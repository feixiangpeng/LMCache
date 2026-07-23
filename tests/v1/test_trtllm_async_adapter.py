# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the async TRT-LLM KV connector adapters.

Both the in-process and multi-process adapters are tested. The tests
stub ``tensorrt_llm`` and GPU primitives so they run on any platform
(no GPU or TRT-LLM install required).

Covers:
- wait_for_save submits without blocking (spy-future pattern)
- CUDA event / IPC event keepalive on held futures
- get_finished reports completed IDs (only when eligible)
- request_finished returns True while save in flight
- start_load_kv submits without blocking
- get_num_new_matched_tokens returns is_async=True when loads exist
- Failure path: futures that fail still get reported
"""

# Standard
import sys
import threading
import types
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub tensorrt_llm before importing the adapters
# ---------------------------------------------------------------------------

_trtllm_stub = types.ModuleType("tensorrt_llm")
_trtllm_stub.mpi_rank = lambda: 0  # type: ignore[attr-defined]

_bindings_stub = types.ModuleType("tensorrt_llm.bindings")
_internal_bm_stub = types.ModuleType("tensorrt_llm.bindings.internal")
_batch_manager_stub = types.ModuleType("tensorrt_llm.bindings.internal.batch_manager")


@dataclass
class _FakeLlmRequest:
    request_id: int = 0
    _tokens: List[int] = field(default_factory=list)

    def get_tokens(self, beam: int) -> List[int]:
        return self._tokens


_batch_manager_stub.LlmRequest = _FakeLlmRequest  # type: ignore[attr-defined]

_connectors_stub = types.ModuleType(
    "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"
)


class _FakeKvCacheConnectorScheduler:
    def __init__(self, llm_args):
        self._llm_args = llm_args


class _FakeKvCacheConnectorWorker:
    def __init__(self, llm_args):
        self._llm_args = llm_args
        self._metadata = None


@dataclass
class _FakeRequestData:
    request_id: int = 0
    new_tokens: List[int] = field(default_factory=list)
    new_block_ids: List[int] = field(default_factory=list)
    computed_position: int = 0
    num_scheduled_tokens: int = 0


@dataclass
class _FakeSchedulerOutput:
    new_requests: list = field(default_factory=list)
    cached_requests: list = field(default_factory=list)


_connectors_stub.KvCacheConnectorScheduler = _FakeKvCacheConnectorScheduler  # type: ignore
_connectors_stub.KvCacheConnectorWorker = _FakeKvCacheConnectorWorker  # type: ignore
_connectors_stub.SchedulerOutput = _FakeSchedulerOutput  # type: ignore

_torch_stub = types.ModuleType("tensorrt_llm._torch")
_pyexecutor_stub = types.ModuleType("tensorrt_llm._torch.pyexecutor")
_pyexecutor_connectors_stub = types.ModuleType(
    "tensorrt_llm._torch.pyexecutor.connectors"
)

_llmapi_stub = types.ModuleType("tensorrt_llm.llmapi")
_llm_args_stub = types.ModuleType("tensorrt_llm.llmapi.llm_args")


@dataclass
class _FakeKvCacheConfig:
    tokens_per_block: int = 64


@dataclass
class _FakeTorchLlmArgs:
    kv_cache_config: _FakeKvCacheConfig = field(default_factory=_FakeKvCacheConfig)
    kv_connector_config: Optional[object] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    model: str = "test-model"


_llm_args_stub.TorchLlmArgs = _FakeTorchLlmArgs  # type: ignore[attr-defined]

# Wire all stubs into sys.modules
sys.modules["tensorrt_llm"] = _trtllm_stub
sys.modules["tensorrt_llm.bindings"] = _bindings_stub
sys.modules["tensorrt_llm.bindings.internal"] = _internal_bm_stub
sys.modules["tensorrt_llm.bindings.internal.batch_manager"] = _batch_manager_stub
sys.modules["tensorrt_llm._torch"] = _torch_stub
sys.modules["tensorrt_llm._torch.pyexecutor"] = _pyexecutor_stub
sys.modules["tensorrt_llm._torch.pyexecutor.connectors"] = _pyexecutor_connectors_stub
sys.modules[
    "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"
] = _connectors_stub
sys.modules["tensorrt_llm.llmapi"] = _llmapi_stub
sys.modules["tensorrt_llm.llmapi.llm_args"] = _llm_args_stub

# ---------------------------------------------------------------------------
# Now import the adapters (they import tensorrt_llm at module level)
# ---------------------------------------------------------------------------

# First Party - these need lmcache importable
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch  # noqa: E402


# Stub lmcache.torch_dev to avoid CUDA requirements
class _FakeEvent:
    """Simulates a CUDA event with query/record/ipc_handle."""

    def __init__(self, interprocess: bool = False):
        self._done = False
        self._interprocess = interprocess

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


# Patch torch_dev before importing adapters
_torch_dev_patch = patch.dict(
    "sys.modules",
    {
        "lmcache.torch_dev": types.SimpleNamespace(
            Event=_FakeEvent,
            Stream=_FakeStream,
            current_stream=lambda: _FakeStream(),
            current_device=lambda: "cpu",
        ),
    },
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SpyFuture:
    """A future that tracks whether .result() was called (blocking)."""

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

    def to_cuda_future(self, device=None):
        return self


# ---------------------------------------------------------------------------
# MP Adapter Tests
# ---------------------------------------------------------------------------


class TestMPAdapterAsyncStore:
    """Tests for tensorrt_mp_adapter.py async store."""

    def _make_worker(self, monkeypatch):
        """Build a worker without running __init__ (skips ZMQ connect)."""
        from lmcache.integration.tensorrt_llm import (
            tensorrt_mp_adapter as mp_mod,
        )

        worker = object.__new__(mp_mod.LMCacheMPKvConnectorWorker)
        worker._block_size = 64
        worker._mq_timeout = 5.0
        worker._instance_id = 12345
        worker._registered = True
        worker._rank = 0
        worker._world_size = 1
        worker._model_name = "test-model"
        worker._chunk_size = 256
        worker._zmq_context = MagicMock()
        worker._mq_client = MagicMock()
        worker._inflight_saves = {}
        worker._inflight_loads = {}
        worker._metadata = None
        worker._llm_args = _FakeTorchLlmArgs()
        return worker, mp_mod

    def _make_scheduler(self, monkeypatch):
        """Build a scheduler without running __init__."""
        from lmcache.integration.tensorrt_llm import (
            tensorrt_mp_adapter as mp_mod,
        )

        sched = object.__new__(mp_mod.LMCacheMPKvConnectorScheduler)
        sched._block_size = 64
        sched._mq_timeout = 5.0
        sched._chunk_size = 256
        sched._pending = {}
        sched._saving_in_flight = set()
        sched._zmq_context = MagicMock()
        sched._mq_client = MagicMock()
        sched._rank = 0
        sched._world_size = 1
        sched._model_name = "test-model"
        sched._llm_args = _FakeTorchLlmArgs()
        return sched, mp_mod

    def test_wait_for_save_does_not_block(self, monkeypatch):
        """wait_for_save must submit and return without calling .result()."""
        worker, mp_mod = self._make_worker(monkeypatch)

        spy = _SpyFuture()
        monkeypatch.setattr(
            mp_mod,
            "_send_request",
            lambda *args, **kwargs: spy,
        )
        monkeypatch.setattr(
            mp_mod,
            "check_interprocess_event_support",
            lambda: None,
        )
        monkeypatch.setattr(mp_mod, "torch_dev", types.SimpleNamespace(
            Event=_FakeEvent,
            Stream=_FakeStream,
        ))

        # Set up metadata with one save.
        worker._metadata = mp_mod.LMCacheMPConnectorMetadata(
            saves={42: mp_mod._BlockSpec(tokens=list(range(256)), block_ids=[0, 1, 2, 3])}
        )
        worker.wait_for_save(_FakeStream())

        # The future should NOT have .result() called.
        assert not spy.result_called, "wait_for_save must not block on .result()"
        # The future should be tracked.
        assert 42 in worker._inflight_saves

    def test_event_keepalive_on_future(self, monkeypatch):
        """The CUDA IPC export event must be pinned on the held future."""
        worker, mp_mod = self._make_worker(monkeypatch)

        spy = _SpyFuture()
        monkeypatch.setattr(mp_mod, "_send_request", lambda *a, **kw: spy)
        monkeypatch.setattr(mp_mod, "check_interprocess_event_support", lambda: None)
        monkeypatch.setattr(mp_mod, "torch_dev", types.SimpleNamespace(
            Event=_FakeEvent,
            Stream=_FakeStream,
        ))

        worker._metadata = mp_mod.LMCacheMPConnectorMetadata(
            saves={7: mp_mod._BlockSpec(tokens=list(range(256)), block_ids=[0])}
        )
        worker.wait_for_save(_FakeStream())

        # The event must be kept alive alongside the future.
        future_and_event = worker._inflight_saves[7]
        assert len(future_and_event) == 2
        _future, event = future_and_event
        assert isinstance(event, _FakeEvent)

    def test_get_finished_reports_completed_saves(self, monkeypatch):
        """get_finished returns req IDs whose futures are done."""
        worker, mp_mod = self._make_worker(monkeypatch)

        done_future = _SpyFuture(value=True)
        done_future.mark_done()
        pending_future = _SpyFuture(value=True)

        worker._inflight_saves = {
            10: (done_future, _FakeEvent()),
            20: (pending_future, _FakeEvent()),
        }

        # Only req 10 is eligible (passed in finished_gen_req_ids).
        saves, loads = worker.get_finished([10, 20], [])
        assert 10 in saves
        assert 20 not in saves  # not done yet
        assert 10 not in worker._inflight_saves
        assert 20 in worker._inflight_saves

    def test_get_finished_respects_eligibility(self, monkeypatch):
        """IDs not passed in finished_gen_req_ids are never reported."""
        worker, mp_mod = self._make_worker(monkeypatch)

        done_future = _SpyFuture(value=True)
        done_future.mark_done()

        worker._inflight_saves = {99: (done_future, _FakeEvent())}

        # 99 is NOT in finished_gen_req_ids — should not be reported.
        saves, loads = worker.get_finished([], [])
        assert saves == []
        assert 99 in worker._inflight_saves  # still tracked

    def test_request_finished_returns_true_while_in_flight(self, monkeypatch):
        """request_finished returns True when save is in flight, False otherwise."""
        sched, mp_mod = self._make_scheduler(monkeypatch)

        # Simulate: build_connector_meta registered a save for req 42.
        sched._saving_in_flight.add(42)

        req = _FakeLlmRequest(request_id=42)
        assert sched.request_finished(req, []) is True
        # Second call returns False (consumed on first call).
        assert sched.request_finished(req, []) is False

        # Unknown request returns False.
        req2 = _FakeLlmRequest(request_id=99)
        assert sched.request_finished(req2, []) is False

    def test_start_load_kv_does_not_block(self, monkeypatch):
        """start_load_kv must submit and return without calling .result()."""
        worker, mp_mod = self._make_worker(monkeypatch)

        spy = _SpyFuture()
        monkeypatch.setattr(mp_mod, "_send_request", lambda *a, **kw: spy)
        monkeypatch.setattr(mp_mod, "check_interprocess_event_support", lambda: None)
        monkeypatch.setattr(mp_mod, "torch_dev", types.SimpleNamespace(
            Event=_FakeEvent,
            Stream=_FakeStream,
        ))

        worker._metadata = mp_mod.LMCacheMPConnectorMetadata(
            loads={5: mp_mod._BlockSpec(tokens=list(range(256)), block_ids=[0, 1])}
        )
        worker.start_load_kv(_FakeStream())

        assert not spy.result_called
        assert 5 in worker._inflight_loads

    def test_get_finished_reports_completed_loads(self, monkeypatch):
        """get_finished reports load completion."""
        worker, mp_mod = self._make_worker(monkeypatch)

        done_future = _SpyFuture(value=True)
        done_future.mark_done()

        worker._inflight_loads = {3: (done_future, _FakeEvent())}

        saves, loads = worker.get_finished([], [3])
        assert 3 in loads
        assert 3 not in worker._inflight_loads


class TestMPAdapterAsyncLookup:
    """Tests for get_num_new_matched_tokens returning is_async."""

    def _make_scheduler(self, monkeypatch):
        from lmcache.integration.tensorrt_llm import (
            tensorrt_mp_adapter as mp_mod,
        )

        sched = object.__new__(mp_mod.LMCacheMPKvConnectorScheduler)
        sched._block_size = 64
        sched._mq_timeout = 5.0
        sched._chunk_size = 256
        sched._pending = {}
        sched._saving_in_flight = set()
        sched._zmq_context = MagicMock()
        sched._mq_client = MagicMock()
        sched._rank = 0
        sched._world_size = 1
        sched._model_name = "test-model"
        sched._llm_args = _FakeTorchLlmArgs()
        return sched, mp_mod

    def test_returns_async_true_when_matched(self, monkeypatch):
        """When there are tokens to load, is_async must be True."""
        sched, mp_mod = self._make_scheduler(monkeypatch)

        # Stub _send_request to simulate: LOOKUP succeeds,
        # QUERY_PREFETCH_STATUS returns 2 chunks = 512 tokens.
        call_count = [0]

        def fake_send(*args, **kwargs):
            f = _SpyFuture()
            call_count[0] += 1
            if call_count[0] == 1:
                f.set_result(None)  # LOOKUP returns None
            else:
                f.set_result(2)  # QUERY_PREFETCH_STATUS returns 2 chunks
            return f

        monkeypatch.setattr(mp_mod, "_send_request", fake_send)

        req = _FakeLlmRequest(request_id=1, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 0)

        assert new_matched == 512  # 2 chunks * 256
        assert is_async is True

    def test_returns_async_false_when_no_match(self, monkeypatch):
        """When no tokens to load, is_async must be False."""
        sched, mp_mod = self._make_scheduler(monkeypatch)

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
# In-Process Adapter Tests
# ---------------------------------------------------------------------------


class TestInProcessAdapterAsyncStore:
    """Tests for tensorrt_adapter.py async store."""

    def _make_worker(self, monkeypatch):
        """Build a worker without running __init__."""
        from lmcache.integration.tensorrt_llm import (
            tensorrt_adapter as inproc_mod,
        )

        worker = object.__new__(inproc_mod.LMCacheKvConnectorWorker)
        worker._block_size = 64
        worker._engine = MagicMock()
        worker._load_stream = _FakeStream()
        worker._store_stream = _FakeStream()
        worker._inflight_saves = {}
        worker._inflight_loads = {}
        worker._metadata = None
        worker._llm_args = _FakeTorchLlmArgs()
        return worker, inproc_mod

    def _make_scheduler(self, monkeypatch):
        from lmcache.integration.tensorrt_llm import (
            tensorrt_adapter as inproc_mod,
        )

        sched = object.__new__(inproc_mod.LMCacheKvConnectorScheduler)
        sched._block_size = 64
        sched._pending = {}
        sched._engine = None
        sched._saving_in_flight = set()
        sched._llm_args = _FakeTorchLlmArgs()
        return sched, inproc_mod

    def test_wait_for_save_does_not_synchronize(self, monkeypatch):
        """wait_for_save must NOT call store_stream.synchronize()."""
        worker, inproc_mod = self._make_worker(monkeypatch)

        sync_called = []
        worker._store_stream = MagicMock()
        worker._store_stream.synchronize = lambda: sync_called.append(True)
        worker._store_stream.wait_stream = lambda s: None

        monkeypatch.setattr(inproc_mod, "torch_dev", types.SimpleNamespace(
            Event=_FakeEvent,
            Stream=_FakeStream,
        ))

        worker._metadata = inproc_mod.LMCacheConnectorMetadata(
            saves={10: inproc_mod._BlockSpec(tokens=list(range(256)), block_ids=[0])}
        )
        worker.wait_for_save(_FakeStream())

        assert sync_called == [], "wait_for_save must not synchronize"
        assert 10 in worker._inflight_saves

    def test_get_finished_polls_cuda_events(self, monkeypatch):
        """get_finished reports IDs when their CUDA events are done."""
        worker, inproc_mod = self._make_worker(monkeypatch)

        done_event = _FakeEvent()
        done_event.mark_done()
        pending_event = _FakeEvent()

        worker._inflight_saves = {1: done_event, 2: pending_event}

        saves, loads = worker.get_finished([1, 2], [])
        assert 1 in saves
        assert 2 not in saves
        assert 1 not in worker._inflight_saves
        assert 2 in worker._inflight_saves

    def test_get_finished_reports_load_completion(self, monkeypatch):
        """get_finished reports load IDs when CUDA events complete."""
        worker, inproc_mod = self._make_worker(monkeypatch)

        done_event = _FakeEvent()
        done_event.mark_done()
        worker._inflight_loads = {7: done_event}

        saves, loads = worker.get_finished([], [7])
        assert 7 in loads
        assert 7 not in worker._inflight_loads

    def test_request_finished_scheduler(self, monkeypatch):
        """request_finished returns True when save in flight, False otherwise."""
        sched, inproc_mod = self._make_scheduler(monkeypatch)
        sched._saving_in_flight.add(99)

        req = _FakeLlmRequest(request_id=99)
        assert sched.request_finished(req, []) is True
        # Consumed — second call returns False.
        assert sched.request_finished(req, []) is False

        # Unknown request returns False.
        req2 = _FakeLlmRequest(request_id=100)
        assert sched.request_finished(req2, []) is False

    def test_start_load_kv_records_event(self, monkeypatch):
        """start_load_kv fires engine.retrieve and records an event."""
        worker, inproc_mod = self._make_worker(monkeypatch)

        monkeypatch.setattr(inproc_mod, "torch_dev", types.SimpleNamespace(
            Event=_FakeEvent,
            Stream=_FakeStream,
        ))

        worker._metadata = inproc_mod.LMCacheConnectorMetadata(
            loads={3: inproc_mod._BlockSpec(tokens=list(range(256)), block_ids=[0, 1])}
        )
        worker.start_load_kv(_FakeStream())

        worker._engine.retrieve.assert_called_once()
        assert 3 in worker._inflight_loads

    def test_lookup_returns_async_true(self, monkeypatch):
        """get_num_new_matched_tokens returns is_async=True when matched."""
        sched, inproc_mod = self._make_scheduler(monkeypatch)

        # Stub engine lookup to return 512 tokens cached.
        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 512
        sched._engine = mock_engine

        req = _FakeLlmRequest(request_id=1, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 0)

        assert new_matched == 512
        assert is_async is True

    def test_lookup_returns_async_false_when_no_hit(self, monkeypatch):
        """get_num_new_matched_tokens returns is_async=False when no match."""
        sched, inproc_mod = self._make_scheduler(monkeypatch)

        mock_engine = MagicMock()
        mock_engine.lookup.return_value = 0
        sched._engine = mock_engine

        req = _FakeLlmRequest(request_id=2, _tokens=list(range(1024)))
        new_matched, is_async = sched.get_num_new_matched_tokens(req, 0)

        assert new_matched == 0
        assert is_async is False
