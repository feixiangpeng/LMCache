# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for the async TRT-LLM KV connector adapters.

The MQ and GPU boundaries are stubbed; no live daemon needed. Covers the
async protocol added for TRT-LLM mode:

- ``wait_for_save`` submits STORE without blocking (spy-future pattern)
- CUDA event / IPC event keepalive on held futures
- ``get_finished`` reports completed IDs (only when eligible per the
  TRT-LLM ABC contract)
- ``request_finished`` returns True exactly once per in-flight save
- ``start_load_kv`` submits RETRIEVE without blocking
- ``get_num_new_matched_tokens`` returns ``is_async=True`` when loads exist

The adapters import ``tensorrt_llm`` at module load; skip cleanly where
it's absent (TRT-LLM is an optional integration, not a hard LMCache
dependency) — same pattern as ``test_sglang_mp_adapter.py``.
"""

# Standard
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock
import threading
import types

# Third Party
import pytest

pytest.importorskip("tensorrt_llm")

# First Party
from lmcache.integration.tensorrt_llm import tensorrt_adapter as inproc_mod
from lmcache.integration.tensorrt_llm import tensorrt_mp_adapter as mp_mod
from lmcache.integration.tensorrt_llm.tensorrt_adapter import (
    LMCacheConnectorMetadata,
    LMCacheKvConnectorScheduler,
    LMCacheKvConnectorWorker,
)
from lmcache.integration.tensorrt_llm.tensorrt_mp_adapter import (
    LMCacheMPConnectorMetadata,
    LMCacheMPKvConnectorScheduler,
    LMCacheMPKvConnectorWorker,
    _BlockSpec,
)

_BLOCK_SIZE = 64
_CHUNK_SIZE = 256


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeLlmRequest:
    """Minimal LlmRequest stand-in: the adapters only touch request_id,
    get_tokens(0), and cache_salt."""

    request_id: int = 0
    _tokens: List[int] = field(default_factory=list)
    cache_salt: str = ""

    def get_tokens(self, beam: int) -> List[int]:
        return self._tokens


class _SpyFuture:
    """Future double that records whether the caller blocked on result()."""

    def __init__(self, value=True):
        self._value = value
        self._done = threading.Event()
        self.result_called = False

    def query(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout=None) -> bool:
        return self._done.wait(timeout)

    def result(self, timeout=None):
        self.result_called = True
        if not self._done.wait(timeout):
            raise TimeoutError
        return self._value

    def set_result(self, value):
        self._value = value
        self._done.set()

    def mark_done(self):
        self._done.set()


class _FakeEvent:
    """CUDA event double with query/record/ipc_handle."""

    def __init__(self, interprocess: bool = False):
        self._done = False

    def record(self, stream=None):
        pass

    def query(self) -> bool:
        return self._done

    def synchronize(self):
        self._done = True

    def ipc_handle(self) -> bytes:
        return b"\x00" * 64

    def mark_done(self):
        self._done = True


class _FakeStream:
    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


_FAKE_TORCH_DEV = types.SimpleNamespace(Event=_FakeEvent, Stream=_FakeStream)


# ---------------------------------------------------------------------------
# Builders (object.__new__ skips __init__: no ZMQ connect, no MPI)
# ---------------------------------------------------------------------------


def _make_mp_worker() -> LMCacheMPKvConnectorWorker:
    worker = object.__new__(LMCacheMPKvConnectorWorker)
    worker._block_size = _BLOCK_SIZE
    worker._mq_timeout = 5.0
    worker._instance_id = 12345
    worker._registered = True
    worker._rank = 0
    worker._world_size = 1
    worker._model_name = "test-model"
    worker._chunk_size = _CHUNK_SIZE
    worker._zmq_context = MagicMock()
    worker._mq_client = MagicMock()
    worker._inflight_saves = {}
    worker._inflight_loads = {}
    worker._eligible_saves = set()
    worker._eligible_loads = set()
    worker._metadata = None
    return worker


def _make_mp_scheduler() -> LMCacheMPKvConnectorScheduler:
    sched = object.__new__(LMCacheMPKvConnectorScheduler)
    sched._block_size = _BLOCK_SIZE
    sched._mq_timeout = 5.0
    sched._chunk_size = _CHUNK_SIZE
    sched._pending = {}
    sched._saving_in_flight = set()
    sched._pending_async_loads = {}
    sched._zmq_context = MagicMock()
    sched._mq_client = MagicMock()
    sched._rank = 0
    sched._world_size = 1
    sched._model_name = "test-model"
    return sched


def _make_inproc_worker() -> LMCacheKvConnectorWorker:
    worker = object.__new__(LMCacheKvConnectorWorker)
    worker._block_size = _BLOCK_SIZE
    worker._engine = MagicMock()
    worker._load_stream = _FakeStream()
    worker._store_stream = _FakeStream()
    worker._inflight_saves = {}
    worker._inflight_loads = {}
    worker._eligible_saves = set()
    worker._eligible_loads = set()
    worker._metadata = None
    return worker


def _make_inproc_scheduler() -> LMCacheKvConnectorScheduler:
    sched = object.__new__(LMCacheKvConnectorScheduler)
    sched._block_size = _BLOCK_SIZE
    sched._pending = {}
    sched._engine = None
    sched._saving_in_flight = set()
    sched._pending_async_loads = {}
    return sched


# ---------------------------------------------------------------------------
# MP adapter: async store
# ---------------------------------------------------------------------------


class TestMPAdapterAsyncStore:
    def test_wait_for_save_does_not_block(self, monkeypatch):
        """wait_for_save must submit and return without calling .result()."""
        worker = _make_mp_worker()

        spy = _SpyFuture()
        monkeypatch.setattr(mp_mod, "_send_request", lambda *a, **kw: spy)
        monkeypatch.setattr(mp_mod, "check_interprocess_event_support", lambda: None)
        monkeypatch.setattr(mp_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheMPConnectorMetadata(
            saves={42: _BlockSpec(tokens=list(range(256)), block_ids=[0, 1, 2, 3])}
        )
        worker.wait_for_save(_FakeStream())

        assert not spy.result_called, "wait_for_save must not block on .result()"
        assert 42 in worker._inflight_saves

    def test_event_keepalive_on_future(self, monkeypatch):
        """The CUDA IPC export event must be pinned alongside the future."""
        worker = _make_mp_worker()

        spy = _SpyFuture()
        monkeypatch.setattr(mp_mod, "_send_request", lambda *a, **kw: spy)
        monkeypatch.setattr(mp_mod, "check_interprocess_event_support", lambda: None)
        monkeypatch.setattr(mp_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheMPConnectorMetadata(
            saves={7: _BlockSpec(tokens=list(range(256)), block_ids=[0])}
        )
        worker.wait_for_save(_FakeStream())

        future_and_event = worker._inflight_saves[7]
        assert len(future_and_event) == 2
        _future, event = future_and_event
        assert isinstance(event, _FakeEvent)

    def test_get_finished_reports_completed_saves(self):
        """get_finished returns req IDs whose futures are done."""
        worker = _make_mp_worker()

        done_future = _SpyFuture(value=True)
        done_future.mark_done()
        pending_future = _SpyFuture(value=True)

        worker._inflight_saves = {
            10: (done_future, _FakeEvent()),
            20: (pending_future, _FakeEvent()),
        }

        saves, loads = worker.get_finished([10, 20], [])
        assert 10 in saves
        assert 20 not in saves  # not done yet
        assert 10 not in worker._inflight_saves
        assert 20 in worker._inflight_saves

    def test_get_finished_respects_eligibility(self):
        """IDs not passed in finished_gen_req_ids are never reported
        (TRT-LLM ABC contract: only return IDs after they've been
        provided in the input args)."""
        worker = _make_mp_worker()

        done_future = _SpyFuture(value=True)
        done_future.mark_done()

        worker._inflight_saves = {99: (done_future, _FakeEvent())}

        saves, loads = worker.get_finished([], [])
        assert saves == []
        assert 99 in worker._inflight_saves  # still tracked

    def test_get_finished_eligibility_is_sticky(self):
        """TRT-LLM passes each ID exactly once. If the operation is still
        pending at that moment, the ID must stay eligible and be reported
        by a LATER get_finished call — regression for the MP async-load
        hang where a slow RETRIEVE was never reported."""
        worker = _make_mp_worker()

        pending_future = _SpyFuture(value=True)
        worker._inflight_loads = {8: (pending_future, _FakeEvent())}

        # Eligibility granted while the future is still pending.
        saves, loads = worker.get_finished([], [8])
        assert loads == []
        assert 8 in worker._inflight_loads

        # Future completes; the next call passes EMPTY args (TRT-LLM
        # only provides each ID once) — must still report it.
        pending_future.mark_done()
        saves, loads = worker.get_finished([], [])
        assert loads == [8]
        assert 8 not in worker._inflight_loads

    def test_get_finished_reports_eligible_save_with_no_future(self):
        """A save ID that becomes eligible but has no in-flight future
        (e.g. the batch was reverted after build_connector_meta registered
        the save, so wait_for_save never fired) must be reported
        immediately — otherwise TRT-LLM defers the request's block
        deallocation forever and the KV pool starves."""
        worker = _make_mp_worker()
        assert worker._inflight_saves == {}

        saves, loads = worker.get_finished([77], [])
        assert saves == [77]

        # Reported once; not again.
        saves, loads = worker.get_finished([], [])
        assert saves == []

    def test_get_finished_fires_end_session_on_save_completion(self, monkeypatch):
        """END_SESSION goes out only after the STORE completes."""
        worker = _make_mp_worker()

        sent: List[tuple] = []

        def _record_send(mq_client, request_type, payloads):
            sent.append((request_type, payloads))
            return _SpyFuture()

        monkeypatch.setattr(mp_mod, "_send_request", _record_send)

        done_future = _SpyFuture(value=True)
        done_future.mark_done()
        worker._inflight_saves = {5: (done_future, _FakeEvent())}

        saves, _loads = worker.get_finished([5], [])
        assert saves == [5]
        assert len(sent) == 1
        req_type, payloads = sent[0]
        assert req_type == mp_mod.RequestType.END_SESSION
        assert payloads == ["5"]

    def test_request_finished_consume_on_read(self):
        """request_finished returns True exactly once per registered save."""
        sched = _make_mp_scheduler()
        sched._saving_in_flight.add(42)

        req = _FakeLlmRequest(request_id=42)
        assert sched.request_finished(req, []) is True
        # Consumed — second call returns False (also prevents unbounded
        # growth of the tracking set over a long-running server).
        assert sched.request_finished(req, []) is False

        req2 = _FakeLlmRequest(request_id=99)
        assert sched.request_finished(req2, []) is False

    def test_start_load_kv_does_not_block(self, monkeypatch):
        """start_load_kv must submit and return without calling .result()."""
        worker = _make_mp_worker()

        spy = _SpyFuture()
        monkeypatch.setattr(mp_mod, "_send_request", lambda *a, **kw: spy)
        monkeypatch.setattr(mp_mod, "check_interprocess_event_support", lambda: None)
        monkeypatch.setattr(mp_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheMPConnectorMetadata(
            loads={5: _BlockSpec(tokens=list(range(256)), block_ids=[0, 1])}
        )
        worker.start_load_kv(_FakeStream())

        assert not spy.result_called
        assert 5 in worker._inflight_loads

    def test_get_finished_reports_completed_loads(self):
        """get_finished reports load completion in the second list."""
        worker = _make_mp_worker()

        done_future = _SpyFuture(value=True)
        done_future.mark_done()

        worker._inflight_loads = {3: (done_future, _FakeEvent())}

        saves, loads = worker.get_finished([], [3])
        assert 3 in loads
        assert 3 not in worker._inflight_loads


# ---------------------------------------------------------------------------
# MP adapter: async lookup
# ---------------------------------------------------------------------------


class TestMPAdapterAsyncLookup:
    def test_returns_async_true_when_matched(self, monkeypatch):
        """When there are tokens to load, is_async must be True."""
        sched = _make_mp_scheduler()

        call_count = [0]

        def fake_send(*args, **kwargs):
            f = _SpyFuture()
            call_count[0] += 1
            if call_count[0] == 1:
                f.set_result(None)  # LOOKUP returns None
            else:
                f.set_result(2)  # QUERY_PREFETCH_STATUS: 2 chunks cached
            return f

        monkeypatch.setattr(mp_mod, "_send_request", fake_send)

        req = _FakeLlmRequest(request_id=1, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 0)

        assert new_matched == 512  # 2 chunks * 256
        assert is_async is True

    def test_returns_async_false_when_no_match(self, monkeypatch):
        """When no tokens to load, is_async must be False."""
        sched = _make_mp_scheduler()

        call_count = [0]

        def fake_send(*args, **kwargs):
            f = _SpyFuture()
            call_count[0] += 1
            if call_count[0] == 1:
                f.set_result(None)
            else:
                f.set_result(0)  # no cached tokens
            return f

        monkeypatch.setattr(mp_mod, "_send_request", fake_send)

        req = _FakeLlmRequest(request_id=2, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 0)

        assert new_matched == 0
        assert is_async is False


# ---------------------------------------------------------------------------
# In-process adapter
# ---------------------------------------------------------------------------


class TestInProcessAdapterAsyncStore:
    def test_wait_for_save_does_not_synchronize(self, monkeypatch):
        """wait_for_save must NOT call store_stream.synchronize()."""
        worker = _make_inproc_worker()

        sync_called = []
        worker._store_stream = MagicMock()
        worker._store_stream.synchronize = lambda: sync_called.append(True)
        worker._store_stream.wait_stream = lambda s: None

        monkeypatch.setattr(inproc_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheConnectorMetadata(
            saves={10: inproc_mod._BlockSpec(tokens=list(range(256)), block_ids=[0])}
        )
        worker.wait_for_save(_FakeStream())

        assert sync_called == [], "wait_for_save must not synchronize"
        assert 10 in worker._inflight_saves

    def test_get_finished_polls_cuda_events(self):
        """get_finished reports IDs when their CUDA events are done."""
        worker = _make_inproc_worker()

        done_event = _FakeEvent()
        done_event.mark_done()
        pending_event = _FakeEvent()

        worker._inflight_saves = {1: done_event, 2: pending_event}

        saves, loads = worker.get_finished([1, 2], [])
        assert 1 in saves
        assert 2 not in saves
        assert 1 not in worker._inflight_saves
        assert 2 in worker._inflight_saves

    def test_get_finished_reports_load_completion(self):
        """get_finished reports load IDs when CUDA events complete."""
        worker = _make_inproc_worker()

        done_event = _FakeEvent()
        done_event.mark_done()
        worker._inflight_loads = {7: done_event}

        saves, loads = worker.get_finished([], [7])
        assert 7 in loads
        assert 7 not in worker._inflight_loads

    def test_request_finished_consume_on_read(self):
        """request_finished returns True exactly once per registered save."""
        sched = _make_inproc_scheduler()
        sched._saving_in_flight.add(99)

        req = _FakeLlmRequest(request_id=99)
        assert sched.request_finished(req, []) is True
        assert sched.request_finished(req, []) is False

        req2 = _FakeLlmRequest(request_id=100)
        assert sched.request_finished(req2, []) is False

    def test_start_load_kv_records_event(self, monkeypatch):
        """start_load_kv fires engine.retrieve and records an event."""
        worker = _make_inproc_worker()

        monkeypatch.setattr(inproc_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheConnectorMetadata(
            loads={3: inproc_mod._BlockSpec(tokens=list(range(256)), block_ids=[0, 1])}
        )
        worker.start_load_kv(_FakeStream())

        worker._engine.retrieve.assert_called_once()
        assert 3 in worker._inflight_loads

    def test_lookup_returns_async_true(self):
        """get_num_new_matched_tokens returns is_async=True when matched."""
        sched = _make_inproc_scheduler()

        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 512
        sched._engine = mock_engine

        req = _FakeLlmRequest(request_id=1, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 0)

        assert new_matched == 512
        assert is_async is True

    def test_lookup_returns_async_false_when_no_hit(self):
        """get_num_new_matched_tokens returns is_async=False when no match."""
        sched = _make_inproc_scheduler()

        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 0
        sched._engine = mock_engine

        req = _FakeLlmRequest(request_id=2, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 0)

        assert new_matched == 0
        assert is_async is False


# ---------------------------------------------------------------------------
# Overlay path: partial TRT-LLM device match + fuller LMCache match
# ---------------------------------------------------------------------------


class TestInProcessOverlayMath:
    """Verify the remainder math when TRT-LLM already matched some tokens.

    When TRT-LLM's own block reuse supplies ``num_computed_tokens`` on
    device and LMCache holds ``cached`` tokens total, the scheduler must
    load only the non-overlapping, block-aligned remainder:
    ``new_matched == align_down(cached - num_computed, block)``. This is
    exercised incidentally in the e2e runs but is asserted deterministically
    here because forcing exact partial device residency from outside
    TRT-LLM's LRU is unreliable.
    """

    def test_overlay_loads_only_remainder(self):
        """cached=576, TRT matched 128 -> load the aligned remainder (448)."""
        sched = _make_inproc_scheduler()
        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 576
        sched._engine = mock_engine

        req = _FakeLlmRequest(request_id=1, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 128)

        assert new_matched == 448  # align_down(576 - 128, 64)
        assert is_async is True

    def test_overlay_result_is_block_aligned(self):
        """A non-block-multiple overlap is floored to the block boundary."""
        sched = _make_inproc_scheduler()
        mock_engine = MagicMock()
        # cached - num_computed = 600 - 128 = 472; align_down(472, 64) = 448.
        mock_engine.lookup.return_value = 600
        sched._engine = mock_engine

        req = _FakeLlmRequest(request_id=2, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 128)

        assert new_matched == 448
        assert is_async is True

    def test_no_overlay_when_trt_covers_lmcache(self):
        """TRT already matched >= what LMCache holds -> nothing new to load."""
        sched = _make_inproc_scheduler()
        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 256
        sched._engine = mock_engine

        req = _FakeLlmRequest(request_id=3, _tokens=list(range(1024)))
        # TRT matched 256; LMCache holds 256 -> remainder 0.
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 256)

        assert new_matched == 0
        assert is_async is False

    def test_short_circuit_when_trt_matched_all_blocks(self):
        """When TRT covers every full block, lookup is skipped entirely."""
        sched = _make_inproc_scheduler()
        mock_engine = MagicMock()
        sched._engine = mock_engine

        # 640 tokens -> 10 full blocks (max_block_aligned = 640). TRT
        # matched all 640 -> scheduler short-circuits before calling lookup.
        req = _FakeLlmRequest(request_id=4, _tokens=list(range(640)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 640)

        assert new_matched == 0
        assert is_async is False
        mock_engine.lookup.assert_not_called()


# ---------------------------------------------------------------------------
# Cache salt isolation tests
# ---------------------------------------------------------------------------


class TestInProcessCacheSaltIsolation:
    """Verify the in-process adapter threads cache_salt into engine calls."""

    def test_lookup_passes_request_configs_with_salt(self):
        """lookup must pass request_configs with lmcache.tag.cache_salt."""
        sched = _make_inproc_scheduler()

        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 512
        sched._engine = mock_engine

        req = _FakeLlmRequest(
            request_id=1, _tokens=list(range(1024)), cache_salt="user-abc"
        )
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 0)

        assert new_matched == 512
        assert is_async is True
        mock_engine.lookup.assert_called_once()
        call_kwargs = mock_engine.lookup.call_args[1]
        assert call_kwargs["request_configs"] == {
            "lmcache.tag.cache_salt": "user-abc"
        }

    def test_lookup_no_request_configs_when_salt_empty(self):
        """When cache_salt is empty, request_configs should be None."""
        sched = _make_inproc_scheduler()

        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 256
        sched._engine = mock_engine

        req = _FakeLlmRequest(request_id=2, _tokens=list(range(1024)))
        sched.get_num_new_matched_tokens(req, 0)

        call_kwargs = mock_engine.lookup.call_args[1]
        assert call_kwargs.get("request_configs") is None

    def test_store_passes_request_configs_with_salt(self, monkeypatch):
        """wait_for_save must pass cache_salt via request_configs."""
        worker = _make_inproc_worker()
        monkeypatch.setattr(inproc_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheConnectorMetadata(
            saves={
                10: inproc_mod._BlockSpec(
                    tokens=list(range(256)),
                    block_ids=[0, 1],
                    cache_salt="tenant-x",
                )
            }
        )
        worker.wait_for_save(_FakeStream())

        worker._engine.store.assert_called_once()
        call_kwargs = worker._engine.store.call_args[1]
        assert call_kwargs["request_configs"] == {
            "lmcache.tag.cache_salt": "tenant-x"
        }

    def test_retrieve_passes_request_configs_with_salt(self, monkeypatch):
        """start_load_kv must pass cache_salt via request_configs."""
        worker = _make_inproc_worker()
        monkeypatch.setattr(inproc_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheConnectorMetadata(
            loads={
                5: inproc_mod._BlockSpec(
                    tokens=list(range(256)),
                    block_ids=[0, 1],
                    cache_salt="tenant-y",
                )
            }
        )
        worker.start_load_kv(_FakeStream())

        worker._engine.retrieve.assert_called_once()
        call_kwargs = worker._engine.retrieve.call_args[1]
        assert call_kwargs["request_configs"] == {
            "lmcache.tag.cache_salt": "tenant-y"
        }

    def test_salt_threaded_through_build_connector_meta(self):
        """cache_salt captured in get_num_new_matched_tokens must appear in
        the _BlockSpec emitted by build_connector_meta."""
        sched = _make_inproc_scheduler()

        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 512
        sched._engine = mock_engine

        req = _FakeLlmRequest(
            request_id=7, _tokens=list(range(1024)), cache_salt="salt-42"
        )
        sched.get_num_new_matched_tokens(req, 0)

        sched_req = MagicMock()
        sched_req.request_id = 7
        sched_req.new_block_ids = list(range(16))
        sched_req.computed_position = 0
        sched_req.new_tokens = list(range(1024))

        sched_output = MagicMock()
        sched_output.new_requests = [sched_req]

        meta = sched.build_connector_meta(sched_output)
        assert meta.loads[7].cache_salt == "salt-42"
        assert meta.saves[7].cache_salt == "salt-42"


class TestMPCacheSaltIsolation:
    """Verify the MP adapter threads cache_salt into IPCCacheServerKey."""

    def test_lookup_key_includes_cache_salt(self, monkeypatch):
        """The LOOKUP key must carry the request's cache_salt."""
        sched = _make_mp_scheduler()

        sent_keys = []

        def fake_send(mq_client, request_type, payloads):
            if request_type == mp_mod.RequestType.LOOKUP:
                sent_keys.append(payloads[0])
            f = _SpyFuture()
            f.set_result(2)
            return f

        monkeypatch.setattr(mp_mod, "_send_request", fake_send)

        req = _FakeLlmRequest(
            request_id=1, _tokens=list(range(1024)), cache_salt="user-abc"
        )
        sched.get_num_new_matched_tokens(req, 0)

        assert len(sent_keys) == 1
        assert sent_keys[0].cache_salt == "user-abc"

    def test_store_key_includes_cache_salt(self, monkeypatch):
        """The STORE key must carry cache_salt from the metadata spec."""
        worker = _make_mp_worker()

        sent_keys = []

        def fake_send(mq_client, request_type, payloads):
            if request_type == mp_mod.RequestType.STORE:
                sent_keys.append(payloads[0])
            f = _SpyFuture()
            f.set_result(True)
            return f

        monkeypatch.setattr(mp_mod, "_send_request", fake_send)
        monkeypatch.setattr(mp_mod, "check_interprocess_event_support", lambda: None)
        monkeypatch.setattr(mp_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheMPConnectorMetadata(
            saves={
                42: _BlockSpec(
                    tokens=list(range(256)),
                    block_ids=[0, 1, 2, 3],
                    cache_salt="tenant-z",
                )
            }
        )
        worker.wait_for_save(_FakeStream())

        assert len(sent_keys) == 1
        assert sent_keys[0].cache_salt == "tenant-z"

    def test_retrieve_key_includes_cache_salt(self, monkeypatch):
        """The RETRIEVE key must carry cache_salt from the metadata spec."""
        worker = _make_mp_worker()

        sent_keys = []

        def fake_send(mq_client, request_type, payloads):
            if request_type == mp_mod.RequestType.RETRIEVE:
                sent_keys.append(payloads[0])
            f = _SpyFuture()
            f.set_result(True)
            return f

        monkeypatch.setattr(mp_mod, "_send_request", fake_send)
        monkeypatch.setattr(mp_mod, "check_interprocess_event_support", lambda: None)
        monkeypatch.setattr(mp_mod, "torch_dev", _FAKE_TORCH_DEV)

        worker._metadata = LMCacheMPConnectorMetadata(
            loads={
                5: _BlockSpec(
                    tokens=list(range(256)),
                    block_ids=[0, 1],
                    cache_salt="tenant-w",
                )
            }
        )
        worker.start_load_kv(_FakeStream())

        assert len(sent_keys) == 1
        assert sent_keys[0].cache_salt == "tenant-w"

    def test_salt_threaded_through_build_connector_meta(self, monkeypatch):
        """cache_salt from get_num_new_matched_tokens reaches the specs."""
        sched = _make_mp_scheduler()

        call_count = [0]

        def fake_send(mq_client, request_type, payloads):
            f = _SpyFuture()
            call_count[0] += 1
            if call_count[0] == 1:
                f.set_result(None)
            else:
                f.set_result(2)
            return f

        monkeypatch.setattr(mp_mod, "_send_request", fake_send)

        req = _FakeLlmRequest(
            request_id=3, _tokens=list(range(1024)), cache_salt="my-salt"
        )
        sched.get_num_new_matched_tokens(req, 0)

        sched_req = MagicMock()
        sched_req.request_id = 3
        sched_req.new_block_ids = list(range(16))
        sched_req.computed_position = 0
        sched_req.new_tokens = list(range(1024))

        sched_output = MagicMock()
        sched_output.new_requests = [sched_req]

        meta = sched.build_connector_meta(sched_output)
        assert meta.loads[3].cache_salt == "my-salt"
        assert meta.saves[3].cache_salt == "my-salt"


# ---------------------------------------------------------------------------
# Abort / cancel cleanup
# ---------------------------------------------------------------------------


class TestAbortCancelCleanup:
    """Cancelled requests must drain the adapters' tracking sets cleanly.

    TRT-LLM has no dedicated abort hook on the connector ABC. Instead,
    ``PyExecutor._handle_canceled_requests`` marks a cancelled request
    finished (``request.finish_by_reason(FinishReason.CANCELLED)``) with
    the explicit intent to "reuse all existing code to clean up the KV
    cache resources" (py_executor.py:6596-6600). A finished request then
    flows through the SAME hooks a normally-finished one does:
    ``_send_kv_async`` calls ``kv_connector_request_finished`` for every
    ``req.is_finished`` (py_executor.py:6048-6050), which drives
    ``request_finished`` on the scheduler and ``get_finished`` on the
    worker every iteration.

    So there is no cancel-specific code in the adapters to exercise; the
    property under test is that the ordinary drain is *complete* under a
    cancel, i.e. no leak (saves) and no early report (loads). These tests
    drive the exact call sequence TRT-LLM issues on cancel and assert the
    tracking sets end empty.
    """

    # -- saves: leak-prevention -------------------------------------------

    def test_inproc_cancel_during_inflight_save_drains(self):
        """Cancel with a completed store: scheduler + worker sets drain.

        The request was registered for saving (build_connector_meta added
        it to ``_saving_in_flight`` and wait_for_save recorded an event).
        On cancel TRT-LLM calls ``request_finished`` (returns True →
        deallocation deferred), then ``get_finished`` reports the ID once
        its CUDA event is done and frees the deferral.
        """
        sched = _make_inproc_scheduler()
        worker = _make_inproc_worker()

        sched._saving_in_flight.add(42)
        done_event = _FakeEvent()
        done_event.mark_done()
        worker._inflight_saves = {42: done_event}

        # request_finished: True exactly once, drains the scheduler set.
        req = _FakeLlmRequest(request_id=42)
        assert sched.request_finished(req, []) is True
        assert 42 not in sched._saving_in_flight

        # get_finished: event done → reported and every set drains.
        saves, loads = worker.get_finished([42], [])
        assert saves == [42]
        assert loads == []
        assert worker._inflight_saves == {}
        assert worker._eligible_saves == set()

    def test_inproc_cancel_during_pending_save_defers_then_drains(self):
        """Cancel with a still-running store must not strand the ID.

        TRT-LLM passes the ID to ``get_finished`` exactly once. If the
        store event is not yet done at that moment, eligibility is sticky:
        the ID stays tracked and is reported by a LATER call (with empty
        args) once the event completes. A cancel must not defeat this.
        """
        sched = _make_inproc_scheduler()
        worker = _make_inproc_worker()

        sched._saving_in_flight.add(7)
        pending_event = _FakeEvent()  # not done
        worker._inflight_saves = {7: pending_event}

        assert sched.request_finished(_FakeLlmRequest(request_id=7), []) is True

        # First poll: eligible but event pending → not reported, still held.
        saves, _loads = worker.get_finished([7], [])
        assert saves == []
        assert 7 in worker._inflight_saves
        assert 7 in worker._eligible_saves

        # Store completes; TRT-LLM passes EMPTY args (ID already provided).
        pending_event.mark_done()
        saves, _loads = worker.get_finished([], [])
        assert saves == [7]
        assert worker._inflight_saves == {}
        assert worker._eligible_saves == set()

    def test_inproc_cancel_of_save_with_no_event_reports_immediately(self):
        """Cancel before wait_for_save ran: the save ID has no event.

        A request can be registered in ``_saving_in_flight`` yet be
        cancelled before the worker recorded a store event (the batch was
        reverted, or cancel landed between build_connector_meta and
        wait_for_save). The eligible-with-no-event fallback must report it
        at once, or TRT-LLM defers its block deallocation forever.
        """
        sched = _make_inproc_scheduler()
        worker = _make_inproc_worker()

        sched._saving_in_flight.add(5)
        assert worker._inflight_saves == {}

        assert sched.request_finished(_FakeLlmRequest(request_id=5), []) is True
        saves, _loads = worker.get_finished([5], [])
        assert saves == [5]
        assert worker._eligible_saves == set()

    def test_mp_cancel_during_inflight_save_drains(self, monkeypatch):
        """MP variant: completed store future drains all sets on cancel."""
        worker = _make_mp_worker()
        sched = _make_mp_scheduler()

        monkeypatch.setattr(mp_mod, "_send_request", lambda *a, **kw: _SpyFuture())

        sched._saving_in_flight.add(42)
        done_future = _SpyFuture(value=True)
        done_future.mark_done()
        worker._inflight_saves = {42: (done_future, _FakeEvent())}

        assert sched.request_finished(_FakeLlmRequest(request_id=42), []) is True
        assert 42 not in sched._saving_in_flight

        saves, loads = worker.get_finished([42], [])
        assert saves == [42]
        assert loads == []
        assert worker._inflight_saves == {}
        assert worker._eligible_saves == set()

    def test_mp_cancel_during_pending_save_defers_then_drains(self, monkeypatch):
        """MP variant: a still-pending store future is not stranded."""
        worker = _make_mp_worker()
        monkeypatch.setattr(mp_mod, "_send_request", lambda *a, **kw: _SpyFuture())

        pending_future = _SpyFuture(value=True)  # not done
        worker._inflight_saves = {7: (pending_future, _FakeEvent())}

        saves, _loads = worker.get_finished([7], [])
        assert saves == []
        assert 7 in worker._inflight_saves
        assert 7 in worker._eligible_saves

        pending_future.mark_done()
        saves, _loads = worker.get_finished([], [])
        assert saves == [7]
        assert worker._inflight_saves == {}
        assert worker._eligible_saves == set()

    # -- loads: corruption-prevention -------------------------------------

    def test_inproc_cancel_during_inflight_load_no_early_report(self):
        """A cancelled, still-loading request must NOT be reported early.

        TRT-LLM moves a request reported as finished-loading back to
        ``CONTEXT_INIT`` to be rescheduled (kv_cache_connector.py:615-617).
        Reporting a load before its retrieve completes would resume decode
        against unloaded KV — silent corruption. Loads therefore have no
        immediate-report fallback, even under cancel. Once the load event
        does complete, the ID drains normally.
        """
        worker = _make_inproc_worker()

        pending_event = _FakeEvent()  # retrieve still running
        worker._inflight_loads = {8: pending_event}

        # Cancel arrives while loading: eligible, but event pending →
        # must stay tracked and unreported.
        saves, loads = worker.get_finished([], [8])
        assert saves == []
        assert loads == []
        assert 8 in worker._inflight_loads

        # Retrieve finishes; the ID drains on the next poll.
        pending_event.mark_done()
        _saves, loads = worker.get_finished([], [])
        assert loads == [8]
        assert worker._inflight_loads == {}
        assert worker._eligible_loads == set()

    def test_mp_cancel_during_inflight_load_no_early_report(self):
        """MP variant: a pending load future is never reported early."""
        worker = _make_mp_worker()

        pending_future = _SpyFuture(value=True)  # not done
        worker._inflight_loads = {8: (pending_future, _FakeEvent())}

        saves, loads = worker.get_finished([], [8])
        assert saves == []
        assert loads == []
        assert 8 in worker._inflight_loads

        pending_future.mark_done()
        _saves, loads = worker.get_finished([], [])
        assert loads == [8]
        assert worker._inflight_loads == {}
        assert worker._eligible_loads == set()

    # -- cancel before the request ever became a save ---------------------

    def test_inproc_cancel_before_save_registered_is_noop(self):
        """Cancel of a request never registered for saving is harmless.

        If a request is cancelled during the async-load phase — before
        build_connector_meta added it to ``_saving_in_flight`` — then
        ``request_finished`` returns False (nothing to defer) and no
        tracking entry is stranded.
        """
        sched = _make_inproc_scheduler()
        worker = _make_inproc_worker()

        req = _FakeLlmRequest(request_id=123)
        assert sched.request_finished(req, []) is False
        assert sched._saving_in_flight == set()

        # And a get_finished naming it as finished-gen still resolves it
        # (no in-flight save → immediate report), leaving nothing behind.
        saves, loads = worker.get_finished([123], [])
        assert saves == [123]
        assert loads == []
        assert worker._inflight_saves == {}
        assert worker._eligible_saves == set()

    def test_mp_cancel_before_save_registered_is_noop(self):
        """MP variant: cancel of an unregistered request drains cleanly."""
        sched = _make_mp_scheduler()
        worker = _make_mp_worker()

        assert sched.request_finished(_FakeLlmRequest(request_id=123), []) is False
        assert sched._saving_in_flight == set()

        saves, loads = worker.get_finished([123], [])
        assert saves == [123]
        assert loads == []
        assert worker._inflight_saves == {}
        assert worker._eligible_saves == set()
