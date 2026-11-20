# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Family routing and the config options the fixed Qwen3.8 kernels refuse to pretend to support."""

import json
from types import SimpleNamespace

import pytest

from pypto_serving.model import model_loader
from pypto_serving.model.model_family import detect_model_family
from pypto_serving.model.qwen3_8_27b.weight_spec import (
    QWEN38_GLOBAL_RULES,
    qwen38_layer_weight_names,
)


def test_hybrid_config_routes_away_from_qwen14b(qwen38_config):
    assert detect_model_family(qwen38_config()) == "qwen3_8_27b"
    assert detect_model_family({"model_type": "qwen3", "hidden_size": 5120}) == "qwen"
    assert detect_model_family({"model_type": "deepseek_v4"}) == "deepseek_v4"


def test_loader_claims_quantized_index_and_keeps_weights_lazy(tmp_path, monkeypatch, qwen38_config):
    config = qwen38_config()
    config["text_config"]["rope_parameters"] = {"rope_type": "default", "rope_theta": 1_000_000.0}
    (tmp_path / "config.json").write_text(json.dumps(config))
    names = {rule.source for rule in QWEN38_GLOBAL_RULES}
    for layer in range(64):
        names.update(qwen38_layer_weight_names(layer))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in names}})
    )
    tokenizer = SimpleNamespace(bos_token_id=None, eos_token_id=None, pad_token_id=None)
    monkeypatch.setattr(model_loader, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    loaded = model_loader.ModelLoader().load("qwen38", str(tmp_path))
    assert loaded.config.head_dim == 256
    assert loaded.config.num_hidden_layers == 64
    assert loaded.config.rope_theta == 1_000_000.0
    assert loaded.runtime_model.extra["family"] == "qwen3_8_27b"
    assert loaded.runtime_model.embed_tokens.numel() == 0
    assert len(loaded.layer_specs) == 64

    config["text_config"]["linear_num_value_heads"] = 32
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="linear_num_value_heads"):
        model_loader.ModelLoader().load("qwen38", str(tmp_path))


@pytest.mark.parametrize(("name", "value", "message"), [
    ("rms_norm_eps", 1e-5, "rms_norm_eps"),
    ("partial_rotary_factor", 0.5, "partial_rotary_factor"),
    ("attention_bias", True, "attention_bias"),
    ("hidden_act", "gelu", "hidden_act"),
    ("layer_types", ["full_attention"] * 64, "layer_types"),
    ("rope_parameters", {"rope_type": "linear"}, "default text RoPE"),
    ("rope_parameters", {"partial_rotary_factor": 0.5}, "partial_rotary_factor"),
    ("rope_parameters", {"rope_theta": 0}, "rope_theta"),
])
def test_loader_rejects_options_not_implemented_by_fixed_kernels(tmp_path, name, value, message,
                                                                 qwen38_config):
    config = qwen38_config()
    config["text_config"][name] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text('{"weight_map":{}}')
    with pytest.raises(ValueError, match=message):
        model_loader.ModelLoader().load("qwen38", str(tmp_path))
