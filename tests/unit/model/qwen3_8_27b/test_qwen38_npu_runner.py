# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host dispatch contracts for the Qwen3.8 runner: descriptors, views and pool geometry.

Every test here drives one launch with ``_run_l3`` replaced, so what the kernel would read
is asserted without a device allocation.
"""

from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.qwen3_8_27b.model_runner import Qwen38ModelRunner
from pypto_serving.model.qwen3_8_27b.npu_runner import Qwen38Layout, resident_view
from pypto_serving.model.qwen3_8_27b.task_args import (
    PREFILL_ARGS,
    DECODE_ARGS,
    decode_task_args,
    embedding_task_args,
    final_head_task_args,
    prefill_task_args,
)
from pypto_serving.model.qwen3_8_27b.weight_spec import (
    QWEN38_ATTENTION_RULES,
    QWEN38_COMMON_RULES,
    QWEN38_GDN_RULES,
)


def test_cache_row_geometry_and_task_arg_order():
    layout = Qwen38Layout(max_seq_len=8192, max_batch_size=16,
                          prefill_tokens=2048, num_kv_pages=16 * 64)
    assert layout.capacity_slots == 16
    assert layout.state_row(1, 0) == 48
    assert layout.state_row(1, 4) == 51
    assert layout.attention_layer_base(7) == layout.kv_layer_stride
    assert layout.conv_pool_shape == ((16 + 1) * 48 + 16, 3, 10240)
    assert layout.recurrent_pool_shape == ((16 + 1) * 48 + 16, 48, 128, 128)
    assert layout.decode_scratch_kv_page(0) == 16 * 64
    assert layout.decode_scratch_state_row(0) == (16 + 1) * 48
    assert prefill_task_args(layout, lambda name: name).names == PREFILL_ARGS
    assert decode_task_args(layout, lambda name: name).names == DECODE_ARGS


@pytest.mark.parametrize("lengths,starts,bases", [
    ([2048], [256], [48]),
    ([256, 1792], [0, 512], [48, 0]),
    ([256] * 8, [256 * row for row in range(8)], [48 * row for row in reversed(range(8))]),
])
def test_prefill_descriptors_address_each_owner_and_layer(lengths, starts, bases):
    rows = len(lengths)
    layout = Qwen38Layout(max_seq_len=4096, max_batch_size=max(2, rows),
                          prefill_tokens=2048, num_kv_pages=32 * max(2, rows))
    runner = object.__new__(Qwen38ModelRunner)
    runner._layout = layout
    task = prefill_task_args(layout, lambda name: runner._active_values[name], rows)
    task.allocate_host_shared(layout)
    buffers = task.tensors
    runner._prefill_tasks = {rows: task}
    runner._compiled = SimpleNamespace(weights={}, prefill_by_rows={rows: "prefill"})
    runner._device = dict.fromkeys(PREFILL_ARGS, "static")
    runner._device["zero_prefill_residual"] = "zero"
    runner._scratch = {"prefill_residual": "residual", "prefill_delta": "delta"}
    runner._cache = SimpleNamespace(conv_state="conv", final_state="final", k_pages="k", v_pages="v")

    def dispatch(kernel, *args):
        assert kernel == "prefill" and len(args) == len(PREFILL_ARGS)
        buffers["logits"].fill_(float("nan"))
        for row in range(rows):
            buffers["logits"][row].fill_(row + 7)

    runner._run_l3 = dispatch
    spans = [torch.arange(length) + 10000 * row for row, length in enumerate(lengths)]
    pages = [list(reversed(range(32 * row, 32 * (row + 1)))) for row in range(rows)]
    logits = runner._run_prefill_chunk(SimpleNamespace(config=SimpleNamespace(rope_theta=1e7)),
                                       spans, starts, pages, bases)
    assert buffers["input_ids"].tolist() == torch.cat(spans).tolist()
    end = 0
    for row, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        base = end
        end += length
        assert torch.all(buffers["row_base"][base:end] == base)
        assert torch.all(buffers["row_owner"][base:end] == row)
        assert buffers["row_ends"][row] == end
        assert buffers["logit_ids"][row] == end - 1
        assert buffers["page_bases"][row] == start // 128
        assert buffers["visible_pages"][row] == (start + length) // 128
        assert buffers["block_table"][row].tolist() == pages[row]
        positions = torch.arange(start, start + length, dtype=torch.float64)
        assert torch.allclose(buffers["rope_cos"][base:end, 0], positions.cos().float())
        assert torch.allclose(buffers["rope_sin"][base:end, 0], positions.sin().float())
    assert buffers["state_rows"].tolist() == [base + layer for layer in range(48) for base in bases]
    assert runner._active_values["conv_state"] == "conv"
    assert runner._active_values["final_state"] == "final"
    assert runner._active_values["valid_rows"] == rows
    assert torch.isfinite(logits).all() and logits[:, 0].tolist() == list(range(7, 7 + rows))
    buffers["logits"].zero_()
    assert logits[0, 0] == 7  # The next launch cannot overwrite returned logits.


def test_resident_view_names_the_requested_byte_range():
    class Buffer:
        def tensor(self, shape, dtype, byte_offset):
            return (tuple(shape), dtype.name, byte_offset)

    device = SimpleNamespace(buffer=Buffer(), shape=(100, 4), dtype=torch.bfloat16)
    assert resident_view(device, 20, (10, 4)) == ((10, 4), "BFLOAT16", 40)
    try:
        resident_view(device, 390, (4, 4))
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("out-of-range resident view was accepted")


def test_full_weight_views_flatten_norms_and_keep_matrix_slabs():
    class Buffer:
        def tensor(self, shape, dtype, byte_offset):
            return (tuple(shape), dtype.name, byte_offset)

    norm = SimpleNamespace(buffer=Buffer(), shape=(64, 3), dtype=torch.bfloat16)
    matrix = object()
    runner = object.__new__(Qwen38ModelRunner)
    runner._compiled = SimpleNamespace(weights={
        "norm_in_w": torch.empty((64, 3), dtype=torch.bfloat16),
        "w_qkv": torch.empty((48 * 2, 3), dtype=torch.int8),
    })
    runner._device = {"norm_in_w": norm, "w_qkv": matrix}
    assert runner._full_weight("norm_in_w") == ((192,), "BFLOAT16", 0)
    assert runner._full_weight("w_qkv") is matrix


def test_decode_weight_views_release_before_their_device_storage():
    runner = object.__new__(Qwen38ModelRunner)
    runner._decode_weight_views = {"norm_in_w": object()}
    runner._device = {"norm_in_w": object()}
    runner._cache = None
    runner._scratch = {}
    runner._reset_l3_dispatch = lambda: None
    freed = []

    def free(tensor):
        assert not runner._decode_weight_views
        freed.append(tensor)

    runner._l3_worker = SimpleNamespace(free_tensor=free, close=lambda: None)
    runner.close()
    assert len(freed) == 1 and not runner._device
    runner.close()


@pytest.mark.parametrize("emit_logits", [True, False])
@pytest.mark.parametrize("scratch", [True, False])
def test_decode_stack_addresses_all_owners_and_padding(emit_logits, scratch):
    layout = Qwen38Layout(max_seq_len=2048, max_batch_size=3,
                          prefill_tokens=2048, num_kv_pages=48)
    runner = object.__new__(Qwen38ModelRunner)
    runner._layout = layout
    runner._cache = SimpleNamespace(k_pages="K", v_pages="V", conv_state="C", final_state="S")
    runner._device = {
        "embed_weight": "embed", "zero_decode_residual": "zero", "tail_mask": "mask",
        "final_norm_w": "norm", "lm_w": "lm",
    }
    runner._scratch = {"decode_delta_a": "da", "decode_delta_b": "db", "decode_residual_b": "rb"}
    names = QWEN38_COMMON_RULES + QWEN38_GDN_RULES + QWEN38_ATTENTION_RULES
    runner._compiled = SimpleNamespace(
        weights={rule.name: object() for rule in names},
        decode_embedding="embed", decode_fwd_by_rows={1: "decode1", 2: "decode2", 16: "decode"},
        final_head="head",
    )
    runner._decode_weight_views = {}
    source = lambda name: runner._active_values[name]
    runner._tasks = {
        "embed": embedding_task_args(layout, source),
        "decode": decode_task_args(layout, source),
        "head": final_head_task_args(layout, source),
    }
    for task in runner._tasks.values():
        task.allocate_host_shared(layout)
    viewed = []

    def weight(name):
        viewed.append(name)
        return (name, "full")

    runner._full_weight = weight
    calls = []

    def dispatch(kernel, *args):
        calls.append(kernel)
        if kernel.startswith("decode"):
            assert len(args) == len(DECODE_ARGS)
            assert runner._active_values["group_start"] == 0
            assert runner._active_values["k_pages"] == "K"
            assert runner._active_values["v_pages"] == "V"
            assert runner._active_values["conv_state"] == "C"
            assert runner._active_values["final_state"] == "S"
        if kernel == "head":
            assert runner._active_values["residual"] == "rb"
            assert runner._active_values["delta"] == "db"
            runner._tasks["head"].tensors["logits"][:2].fill_(42)

    runner._run_l3 = dispatch
    model = SimpleNamespace(config=SimpleNamespace(rope_theta=10_000_000.0))
    logits = runner._run_decode_tokens(model, torch.tensor([11, 12]), [129, 1],
                                        [[7, 8], [9]], [1, 0], scratch=scratch,
                                        emit_logits=emit_logits)
    assert calls == ["embed", "decode2"] + (["head"] if emit_logits else [])
    buffers = runner._tasks["decode"].tensors
    states = buffers["state_rows"].view(48, layout.decode_batch_pad)
    for local in range(48):
        expected = [layout.decode_scratch_state_row(row) if scratch or row >= 2
                    else (48 + local if row == 0 else local) for row in range(layout.decode_batch_pad)]
        assert states[local].tolist() == expected
    tables = buffers["block_tables"].view(16, layout.decode_batch_pad, -1)
    for local, table in enumerate(tables):
        base = local * layout.kv_layer_stride // 128
        assert table[0, :2].tolist() == [base + 7, base + 8]
        assert table[1, 0] == base + 9
        assert table[2:, 0].tolist() == [base + layout.decode_scratch_kv_page(row)
                                      for row in range(2, layout.decode_batch_pad)]
        assert int(table.max()) < layout.kv_pool_shape[0] // 128
    if emit_logits:
        assert logits.shape == (2, 248320) and torch.all(logits == 42)
    else:
        assert logits is None
    expected_views = {name for name in DECODE_ARGS if name in runner._compiled.weights}
    assert set(viewed) == expected_views and len(viewed) == len(expected_views)

    # A narrower next batch must overwrite every old descriptor, retaining only static weight views.
    runner._run_decode_tokens(model, torch.tensor([13]), [1], [[10]], [0], emit_logits=False)
    assert calls[-2:] == ["embed", "decode1"]
    assert len(viewed) == len(expected_views)
    assert buffers["seq_lens"].tolist() == [1] * layout.decode_batch_pad
    for local in range(48):
        assert int(states[local, 0]) == local
        assert states[local, 1:].tolist() == [layout.decode_scratch_state_row(row)
                                            for row in range(1, layout.decode_batch_pad)]
    for local, table in enumerate(tables):
        base = local * layout.kv_layer_stride // 128
        assert int(table[0, 0]) == base + 10
        assert torch.all(table[0, 1:] == base)
        assert int(table[1, 0]) == base + layout.decode_scratch_kv_page(1)

    runner._run_decode_tokens(model, torch.tensor([14, 15, 16]), [1, 1, 1],
                              [[11], [12], [13]], [0, 1, 2], emit_logits=False)
    assert calls[-2:] == ["embed", "decode"]
