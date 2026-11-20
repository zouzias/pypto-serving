# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Greedy generation on one A2/A3 NPU with the Eco-Tech Qwen3.8-27B W8A8 checkpoint.

Set PYPTO_QWEN38_MODEL_DIR to the checkpoint and run through task-submit.
"""

import asyncio
import os
from pathlib import Path

import pytest


MAX_BATCH_SIZE = 2
PREFILL_TOKENS = 2048
MAX_NEW_TOKENS = 8
PROMPT = "The capital of France is"
EXPECTED_TOKEN_IDS = [11751, 13, 198, 760, 6511, 314, 9564, 369]


def qwen38_device_settings():
    model_dir = os.environ.get("PYPTO_QWEN38_MODEL_DIR")
    if not model_dir or not Path(model_dir).is_dir():
        pytest.fail("PYPTO_QWEN38_MODEL_DIR must point to the Eco-Tech Qwen3.8-27B W8A8 checkpoint")
    device = os.environ.get("DEVICE_ID") or os.environ.get("TASK_DEVICE")
    if not device or not device.isdecimal():
        pytest.fail("DEVICE_ID or TASK_DEVICE must identify one A2/A3 NPU")
    return model_dir, device


def make_qwen38_engine():
    model_dir, device = qwen38_device_settings()

    from pypto_serving.cli.main import build_parser, build_serving_engine_config
    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

    args = build_parser().parse_args([
        "--model", model_dir, "--device", device, "--platform", "a2a3",
        "--block-size", "128", "--max-model-len", "4096",
        "--max-num-seqs", str(MAX_BATCH_SIZE),
        "--max-num-batched-tokens", str(PREFILL_TOKENS),
        "--long-prefill-token-threshold", str(PREFILL_TOKENS),
        # GDN state is mutable and belongs to one request.
        "--no-enable-prefix-caching",
    ])
    config = build_serving_engine_config(args)
    return AsyncLLMEngine(config=config, tokenizer=load_tokenizer(model_dir))


def greedy_config(*, stream=False):
    from pypto_serving.config.types import GenerateConfig

    return GenerateConfig(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, stream=stream)


def test_qwen38_output_matches_expected_tokens():
    async def run():
        engine = make_qwen38_engine()
        await engine.start()
        try:
            return await engine.generate_result(PROMPT, greedy_config())
        finally:
            await engine.stop()

    result = asyncio.run(run())
    assert result.token_ids == EXPECTED_TOKEN_IDS, result
    assert result.finish_reason == "length", result
