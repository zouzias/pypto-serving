# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3.8 serving guards for concurrent decode and fused prefill with a tail."""

import asyncio

import pytest

from tests.test_qwen38_accuracy import (
    EXPECTED_TOKEN_IDS,
    MAX_BATCH_SIZE,
    PREFILL_TOKENS,
    PROMPT,
    greedy_config,
    make_qwen38_engine,
)


SECOND_PROMPT = "The capital of Germany is"
SECOND_EXPECTED_TOKEN_IDS = [19241, 13, 198, 760, 6511, 314, 9564, 369]
LONG_PROMPT = "The context is background information.\n" * 292 + PROMPT
LONG_EXPECTED_TOKEN_IDS = [11751, 13, 271, 27775, 383, 279, 3766, 2193]


async def _assert_stream_tokens(engine, prompt, expected_token_ids):
    tokens = []
    final = None
    request_id = engine.generate_request_id()
    async for output in engine.add_request(request_id, prompt, greedy_config(stream=True)):
        if output.token_id is not None:
            tokens.append(output.token_id)
        if output.finished:
            final = output
    assert final is not None
    assert engine.normalize_finish_reason(final.finish_reason) == "length", final
    assert tokens == expected_token_ids, final
    assert tuple(tokens) == final.token_ids, final


@pytest.fixture(scope="module")
def serving():
    engine = make_qwen38_engine()
    assert engine.config.resolve_async_scheduling()
    schedule_events = []
    schedule = engine.scheduler.schedule

    def record_schedule(*args, **kwargs):
        output = schedule(*args, **kwargs)
        if output.scheduled_requests:
            schedule_events.append([
                (request.is_prefill, request.num_new_tokens)
                for request in output.scheduled_requests
            ])
        return output

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine.scheduler, "schedule", record_schedule)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.start())
            try:
                yield engine, loop.run_until_complete, schedule_events
            finally:
                loop.run_until_complete(engine.stop())
        finally:
            loop.close()


def test_qwen38_concurrent_decode(serving):
    engine, run, events = serving
    events.clear()

    async def run_both():
        await asyncio.gather(
            _assert_stream_tokens(engine, PROMPT, EXPECTED_TOKEN_IDS),
            _assert_stream_tokens(engine, SECOND_PROMPT, SECOND_EXPECTED_TOKEN_IDS),
        )

    run(run_both())
    assert any(sum(not is_prefill for is_prefill, _ in step) == MAX_BATCH_SIZE for step in events), events


def test_qwen38_chunked_prefill(serving):
    engine, run, events = serving
    assert len(engine.tokenizer.encode(LONG_PROMPT)) == PREFILL_TOKENS + 1
    events.clear()

    run(_assert_stream_tokens(engine, LONG_PROMPT, LONG_EXPECTED_TOKEN_IDS))

    prefill_sizes = [size for step in events for is_prefill, size in step if is_prefill]
    assert prefill_sizes == [PREFILL_TOKENS, 1], prefill_sizes
