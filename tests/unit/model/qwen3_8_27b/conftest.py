# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint-shaped fixtures shared by the Qwen3.8-27B host-side tests.

Both the loader tests and the executor test need the published ``config.json`` body, and
both the weight-spec test and the executor test need a layer tensor small enough that a
64-layer stack fits in host memory. Keeping them here means a checkpoint contract change
is edited once instead of drifting between files.
"""

import pytest
import torch


@pytest.fixture
def qwen38_config():
    """Return a factory for the hybrid config body, so a test can mutate its own copy."""

    def _build():
        return {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "linear_num_value_heads": 48,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "max_position_embeddings": 262144,
            },
        }

    return _build


@pytest.fixture
def qwen38_tiny_tensor():
    """Return ``(name, layer) -> tensor``: real dtypes and rank, two elements per axis."""

    def _tiny_tensor(name: str, layer: int) -> torch.Tensor:
        suffix = name.split(f"layers.{layer}.", 1)[-1]
        value = float(layer + 1)
        if suffix.endswith("conv1d.weight"):
            return torch.full((2, 1, 3), value, dtype=torch.bfloat16)
        if suffix.endswith("in_proj_a.weight") or suffix.endswith("in_proj_b.weight"):
            return torch.full((1, 2), value, dtype=torch.bfloat16)
        if suffix.endswith("weight_scale"):
            return torch.full((2, 1), value)
        if suffix.endswith("layernorm.weight") or suffix.endswith("norm.weight"):
            return torch.full((2,), value, dtype=torch.bfloat16)
        if suffix.endswith("A_log") or suffix.endswith("dt_bias"):
            return torch.full((2,), value, dtype=torch.bfloat16)
        return torch.full((2, 2), int(layer + 1), dtype=torch.int8)

    return _tiny_tensor
