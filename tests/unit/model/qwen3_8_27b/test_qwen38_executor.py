# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-entry build-out, pypto-lib signature parity, and import isolation for the executor."""

import ast
from types import SimpleNamespace

import torch

from pypto_serving.model import model_loader
from pypto_serving.model.qwen3_8_27b import npu_executor as executor_module
from pypto_serving.model.qwen3_8_27b.cache import build_qwen38_cache_group_specs
from pypto_serving.model.qwen3_8_27b.npu_runner import Qwen38Layout
from pypto_serving.model.qwen3_8_27b.task_args import (
    DECODE_ARGS,
    EMBEDDING_ARGS,
    FINAL_HEAD_ARGS,
    PREFILL_ARGS,
)
from pypto_serving.model.qwen3_8_27b.weight_spec import qwen38_layer_weight_names


def test_executor_builds_all_kernel_entries_without_loading_full_checkpoint(monkeypatch,
                                                                            qwen38_tiny_tensor):
    class TinyStore:
        def __init__(self, **_kwargs):
            pass

        def load_many(self, names):
            result = {}
            for name in names:
                if "layers." in name:
                    layer = int(name.split("layers.")[1].split(".")[0])
                    result[name] = qwen38_tiny_tensor(name, layer)
                elif name.endswith("norm.weight"):
                    result[name] = torch.ones(2, dtype=torch.bfloat16)
                else:
                    result[name] = torch.ones(4, 2, dtype=torch.bfloat16)
            return result

    class Compiler:
        def __init__(self, *, run_config):
            assert run_config.save_kernels_dir is None
            self.names = []

        def compile(self, name, kernel):
            assert kernel is not None
            self.names.append(name)
            return name

    monkeypatch.setattr(executor_module, "Qwen38WeightStore", TinyStore)
    monkeypatch.setattr(executor_module, "KernelCompiler", Compiler)
    executor = executor_module.Qwen38PyptoExecutor()
    compiler = executor._compiler
    runtime = model_loader.RuntimeConfig(
        page_size=128, max_seq_len=2048, max_batch_size=16,
        max_num_batched_tokens=2048, max_prefill_tokens_per_request=2048,
        kv_cache_groups=build_qwen38_cache_group_specs(2048),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=5120, vocab_size=248320,
                               num_hidden_layers=64, head_dim=256),
        runtime=runtime,
        extra={"family": "qwen3_8_27b", "model_dir": "unused", "weight_map": {}},
    )
    compiled = executor._compile_model(model)
    assert set(compiled.prefill_by_rows) == set(range(1, 9))
    assert compiler.names[:8] == [
        f"qwen38_prefill_text_packed_t2048_r{rows}_p16_k34816_s832" for rows in range(1, 9)
    ]
    assert compiler.names[8:] == [
        "qwen38_decode_fwd_b16_g16_p16_k557056_s832",
        "qwen38_decode_fwd_b16_g16_p16_k557056_s832_r1_h2",
        "qwen38_decode_fwd_b16_g16_p16_k557056_s832_r2_h2",
        "qwen38_decode_embedding", "qwen38_decode_final_head",
    ]
    assert set(compiled.decode_fwd_by_rows) == {1, 2, 16}
    assert compiled.layout.kv_pool_shape == (16 * (16 * 16 + 16) * 128, 1024)
    assert compiled.weights["decode_w_qkv"].shape == (16 * 6, 2)
    assert set(compiled.globals) == {"embed_weight", "final_norm_w", "lm_w"}


def test_rule_names_and_task_signatures_match_pypto_lib():
    kernel_dir = executor_module.find_qwen38_kernel_dir()
    root = kernel_dir.parents[1]

    import sys

    sys.path.insert(0, str(root))
    try:
        from models.qwen3_8_27b.contract import layer_weight_specs

        for layer in range(64):
            expected = {spec.name for spec in layer_weight_specs(layer) if not spec.is_zero_offset}
            assert set(qwen38_layer_weight_names(layer)) == expected
    finally:
        sys.path.pop(0)

    sources = {
        "prefill_text_fwd.py": {"qwen3_prefill_text_fwd": PREFILL_ARGS, "l3_prefill_fwd": PREFILL_ARGS},
        "decode_fwd.py": {"qwen3_decode_fwd": DECODE_ARGS, "l3_decode_fwd": DECODE_ARGS},
        "embedding.py": {"_qwen3_embedding": EMBEDDING_ARGS, "l3_embedding": EMBEDDING_ARGS},
        "rms_lm_head.py": {"_qwen3_rms_lm_head": FINAL_HEAD_ARGS, "l3_rms_lm_head": FINAL_HEAD_ARGS},
    }
    for filename, expected_functions in sources.items():
        tree = ast.parse((kernel_dir / filename).read_text())
        found = {
            node.name: tuple(arg.arg for arg in node.args.args)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in expected_functions
        }
        assert found == expected_functions


def test_kernel_capabilities_match_dispatch_geometry():
    contract = executor_module.load_qwen38_serving_contract()
    layout = Qwen38Layout(4096, 2, 2048, 64)
    kv, _ = build_qwen38_cache_group_specs(layout.max_seq_len)
    assert contract.kv_page_tokens == kv.spec.block_size
    assert contract.decode_batch_tile == layout.decode_batch_pad
    assert contract.prefill_tile_tokens == layout.prefill_tokens
    assert contract.prefill_request_alignment * Qwen38Layout(4096, 16, 2048, 512).max_prefill_rows == 2048


def test_kernel_imports_restore_other_models_on_success_and_failure(tmp_path, monkeypatch):
    import sys

    other_config = SimpleNamespace(marker="other-model")
    monkeypatch.setitem(sys.modules, "config", other_config)
    monkeypatch.delitem(sys.modules, "cleanup_test_entry", raising=False)
    (tmp_path / "config.py").write_text("marker = 'qwen'\n")
    (tmp_path / "cleanup_test_entry.py").write_text("import config\nmarker = config.marker\n")
    original_path = list(sys.path)
    for fail in (False, True):
        try:
            with executor_module._qwen38_import_context(tmp_path):
                entry = executor_module.load_qwen38_kernel_module("cleanup_test_entry", tmp_path)
                assert entry.marker == "qwen"
                if fail:
                    raise RuntimeError("compile failed")
        except RuntimeError as exc:
            assert str(exc) == "compile failed"
        assert sys.modules["config"] is other_config
        assert "cleanup_test_entry" not in sys.modules
        assert sys.path == original_path
