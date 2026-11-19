# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill scheduling above the kernels: chunking, cache ownership and state resets.

``_run_prefill_chunk`` and ``_run_decode_tokens`` are recorded rather than run, so these
tests assert which launches a batch turns into -- their descriptor contents are covered by
``test_qwen38_npu_runner.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from pypto_serving.config.types import PrefillBatch
from pypto_serving.model.qwen3_8_27b.model_runner import Qwen38ModelRunner
from pypto_serving.model.qwen3_8_27b.npu_runner import Qwen38Layout


def _recording_runner(layout):
    runner = object.__new__(Qwen38ModelRunner)
    runner._layout = layout
    calls = []
    runner._logit_requests = []
    runner._zero_state_range = lambda first, rows: calls.append(("zero", first, rows))

    def prefill(_model, spans, starts, pages, bases):
        calls.append(("prefill", [span.tolist() for span in spans], starts,
                      [row[0] for row in pages], bases))
        return torch.stack([span[-1:].float() for span in spans])

    def decode(_model, tokens, lengths, pages, states, *, emit_logits=True):
        calls.append(("decode", tokens.tolist(), lengths, [row[0] for row in pages], states))
        runner._logit_requests.append((list(states), emit_logits))
        return tokens.float().unsqueeze(1) if emit_logits else None

    runner._run_prefill_chunk = prefill
    runner._run_decode_tokens = decode
    return runner, calls


def _batch(lengths, starts=None, order=None):
    starts = starts or [0] * len(lengths)
    order = list(range(len(lengths))) if order is None else order
    spans = [torch.arange(10000 * row, 10000 * row + length) for row, length in enumerate(lengths)]
    offsets = [sum(lengths[:row]) for row in range(len(lengths))]
    return PrefillBatch(
        request_ids=[str(row) for row in order], token_ids=torch.cat(spans), input_embeddings=None,
        seq_lens=[starts[row] + lengths[row] for row in order],
        chunk_lens=[lengths[row] for row in order], chunk_starts=[starts[row] for row in order],
        chunk_offsets=[offsets[row] for row in order],
        block_ids_by_group=[{"attn_kv": list(range(32 * row, 32 * (row + 1))),
                             "gdn_state": [row]} for row in order],
    )


def test_prefill_uses_full_chunk_then_exact_token_tail():
    layout = Qwen38Layout(max_seq_len=4096, max_batch_size=1,
                          prefill_tokens=2048, num_kv_pages=32)
    runner, calls = _recording_runner(layout)
    result = runner.run_prefill(SimpleNamespace(), _batch([2050]))
    assert calls == [
        ("zero", 0, 48), ("prefill", [list(range(2048))], [0], [0], [0]),
        ("decode", [2048], [2049], [0], [0]), ("decode", [2049], [2050], [0], [0]),
    ]
    assert result.logits.tolist() == [[2049.0]]
    assert runner._logit_requests == [([0], False), ([0], True)]


def test_prefill_packs_mixed_owners_and_routes_logits_in_request_order():
    layout = Qwen38Layout(max_seq_len=4096, max_batch_size=3,
                          prefill_tokens=2048, num_kv_pages=96)
    runner, calls = _recording_runner(layout)
    batch = _batch([256, 1792, 2], starts=[0, 512, 5], order=[2, 1, 0])
    batch.kv_allocations = [SimpleNamespace(tokens_used=0) for _ in range(3)]
    result = runner.run_prefill(SimpleNamespace(), batch)
    assert calls == [
        ("zero", 0, 48),
        ("prefill", [list(range(10000, 11792)), list(range(256))], [512, 0], [32, 0], [48, 0]),
        ("decode", [20000], [6], [64], [2]),
        ("decode", [20001], [7], [64], [2]),
    ]
    assert result.logits.tolist() == [[20001.0], [11791.0], [255.0]]
    assert [allocation.tokens_used for allocation in batch.kv_allocations] == [7, 2304, 256]


def test_prefill_batches_unequal_tails_without_mixing_cache_owners():
    layout = Qwen38Layout(max_seq_len=4096, max_batch_size=2,
                          prefill_tokens=2048, num_kv_pages=64)
    runner, calls = _recording_runner(layout)
    result = runner.run_prefill(SimpleNamespace(), _batch([2050, 2049]))
    assert calls[-2:] == [
        ("decode", [2048, 12048], [2049, 2049], [0, 32], [0, 1]),
        ("decode", [2049], [2050], [0], [0]),
    ]
    assert result.logits.tolist() == [[2049.0], [12048.0]]
    assert runner._logit_requests[-2:] == [([0, 1], True), ([0], True)]


def test_prefill_keeps_aligned_owner_while_other_reaches_boundary():
    layout = Qwen38Layout(max_seq_len=4096, max_batch_size=2,
                          prefill_tokens=2048, num_kv_pages=64)
    runner, calls = _recording_runner(layout)
    result = runner.run_prefill(SimpleNamespace(), _batch([1536, 513], starts=[0, 255]))
    assert calls == [
        ("zero", 0, 48), ("decode", [10000], [256], [32], [1]),
        ("prefill", [list(range(1536)), list(range(10001, 10513))], [0, 256], [0, 32], [0, 48]),
    ]
    assert result.logits.tolist() == [[1535.0], [10512.0]]
    assert runner._logit_requests == [([1], False)]


@pytest.mark.parametrize("lengths,starts", [
    ([256, 1792], [256, 512]),
    ([768, 3840], [0, 0]),
    ([256] * 8, [0] * 8),
    ([17, 255, 257], [3, 0, 511]),
    ([2049, 2307, 1793], [255, 0, 256]),
])
def test_prefill_consumes_every_real_token_once_in_owner_order(lengths, starts):
    layout = Qwen38Layout(max_seq_len=4096, max_batch_size=len(lengths),
                          prefill_tokens=2048, num_kv_pages=32 * len(lengths))
    runner, calls = _recording_runner(layout)
    result = runner.run_prefill(SimpleNamespace(), _batch(lengths, starts))
    seen = [[] for _ in lengths]
    for kind, *args in calls:
        if kind == "zero":
            continue
        if kind == "prefill":
            spans, positions, pages, bases = args
            assert sum(len(span) for span in spans) == 2048
            for span, position, page, base in zip(spans, positions, pages, bases, strict=True):
                owner = base // 48
                assert page == 32 * owner
                assert position == starts[owner] + len(seen[owner])
                assert position % 256 == len(span) % 256 == 0
                seen[owner].extend(span)
        else:
            tokens, positions, pages, owners = args
            for token, position, page, owner in zip(tokens, positions, pages, owners, strict=True):
                assert page == 32 * owner
                assert position == starts[owner] + len(seen[owner]) + 1
                seen[owner].append(token)
    assert seen == [list(range(10000 * row, 10000 * row + length)) for row, length in enumerate(lengths)]
    assert result.logits[:, 0].tolist() == [float(span[-1]) for span in seen]


@pytest.mark.parametrize("conflict", ["state", "write", "prefix", "duplicate", "metadata"])
def test_prefill_rejects_ambiguous_cache_ownership_before_any_dispatch(conflict):
    layout = Qwen38Layout(max_seq_len=4096, max_batch_size=2,
                          prefill_tokens=2048, num_kv_pages=64)
    runner, calls = _recording_runner(layout)
    batch = _batch([256, 1792], starts=[0, 512])
    if conflict == "state":
        batch.block_ids_by_group[1]["gdn_state"] = [0]
    elif conflict == "write":
        batch.block_ids_by_group[1]["attn_kv"][4] = 0
    elif conflict == "prefix":
        batch.block_ids_by_group[1]["attn_kv"][0] = 0
    elif conflict == "duplicate":
        batch.block_ids_by_group[1]["attn_kv"][1] = 32
    else:
        batch.chunk_offsets = [0]
    with pytest.raises(ValueError):
        runner.run_prefill(SimpleNamespace(), batch)
    assert calls == []


def test_prefill_allows_shared_read_only_attention_prefix():
    layout = Qwen38Layout(max_seq_len=4096, max_batch_size=2,
                          prefill_tokens=2048, num_kv_pages=64)
    runner, calls = _recording_runner(layout)
    batch = _batch([256, 1792], starts=[256, 256])
    batch.block_ids_by_group[1]["attn_kv"][:2] = [0, 1]
    result = runner.run_prefill(SimpleNamespace(), batch)
    assert len(calls) == 1 and calls[0][0] == "prefill"
    assert result.logits.tolist() == [[255.0], [11791.0]]


@pytest.mark.parametrize("first,rows", [(-1, 48), (0, 0), (0, 96), (120, 48)])
def test_state_reset_rejects_out_of_bounds_copy_before_worker_dispatch(first, rows):
    runner = object.__new__(Qwen38ModelRunner)
    runner._cache = SimpleNamespace(conv_state=SimpleNamespace(shape=(160, 3, 10240)),
                                    final_state=SimpleNamespace(shape=(160, 48, 128, 128)))
    runner._zero_conv_rows = SimpleNamespace(shape=(48, 3, 10240))
    runner._zero_state_rows = SimpleNamespace(shape=(48, 48, 128, 128))
    runner._shared_l3_worker = lambda: pytest.fail("invalid reset reached the worker")
    with pytest.raises(ValueError, match="State reset exceeds"):
        runner._zero_state_range(first, rows)
