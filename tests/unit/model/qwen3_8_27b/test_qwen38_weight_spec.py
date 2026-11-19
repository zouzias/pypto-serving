# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint-to-slab packing: the stacked layout the fused kernels read as an input contract."""

import pytest
import torch

from pypto_serving.model.qwen3_8_27b.weight_spec import (
    QWEN38_COMMON_RULES,
    pack_qwen38_layer,
    qwen38_layer_weight_names,
    stage_qwen38_weights,
)


@pytest.mark.parametrize("workers", [1, 4])
def test_type_local_stacks_and_decode_projection_fusion(workers, qwen38_tiny_tensor):
    class TinyStore:
        def load_many(self, names):
            return {name: qwen38_tiny_tensor(name, int(name.split("layers.")[1].split(".")[0]))
                    for name in names}

    store = TinyStore()
    packed = pack_qwen38_layer(3, store.load_many(qwen38_layer_weight_names(3)))
    assert packed["decode_w_qkv"].shape == (6, 2)
    assert packed["decode_w_qkv_scale"].shape == (1, 6)
    assert torch.equal(packed["decode_w_qkv"][:2], packed["w_q"])

    slabs = stage_qwen38_weights(store, workers=workers)
    assert slabs["w_qkv"].shape == (48 * 2, 2)
    assert slabs["w_q"].shape == (16 * 2, 2)
    assert slabs["w_gate"].shape == (64 * 2, 2)
    assert slabs["conv_w"].shape == (48 * 3, 2)
    assert slabs["w_ab"].shape == (48 * 2, 2)
    assert slabs["w_qkv"][0, 0].item() == 1
    assert slabs["w_qkv"][-1, 0].item() == 63
    assert slabs["w_q"][0, 0].item() == 4
    assert slabs["w_q"][-1, 0].item() == 64
    assert slabs["w_gate"][-1, 0].item() == 64
    for rule in QWEN38_COMMON_RULES:
        by_layer = slabs[rule.name].reshape(64, -1)
        expected = torch.arange(1, 65, dtype=by_layer.dtype)[:, None].expand_as(by_layer)
        assert torch.equal(by_layer, expected), rule.name
