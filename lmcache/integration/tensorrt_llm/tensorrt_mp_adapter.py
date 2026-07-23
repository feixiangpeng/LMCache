# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM KV Cache Connector adapter for LMCache (multi-process mode).

Implements ``LMCacheMPKvConnectorScheduler`` and
``LMCacheMPKvConnectorWorker`` — the two classes TRT-LLM's
``kv_connector_config`` requires — backed by a standalone LMCache server
reached over ZMQ. Provides process isolation and shared caching across
multiple TRT-LLM instances on the same node.

The KV pool tensor is shared with the server via :class:`RawCudaIPCWrapper`
because TRT-LLM's pool is allocated outside PyTorch's caching allocator
(``at::for_blob`` over ``cudaMalloc``), which makes
``UntypedStorage._share_cuda_()`` raise. The wrapper bypasses that path.

Async protocol:
    - ``wait_for_save`` submits STORE requests and returns immediately,
      holding the futures (+ CUDA IPC event keepalives).
    - ``get_finished`` polls held futures each step and reports completed
      request IDs to the TRT-LLM runtime.
    - ``request_finished`` returns True while a save is in flight,
      deferring GPU block deallocation until completion.
    - ``start_load_kv`` submits RETRIEVE requests without blocking.
    - ``get_num_new_matched_tokens`` returns ``is_async=True`` to park
      the request until loading finishes (reported via ``get_finished``).
"""

# Standard
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import os
import time

# Third Party
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
import torch
import zmq

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType, check_interprocess_event_support
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.platform.cuda.ipc_wrapper import RawCudaIPCWrapper

logger = init_logger(__name__)

DEFAULT_SERVER_URL = "ipc:///tmp/lmcache.sock"
DEFAULT_MQ_TIMEOUT: float = 300.0


def _get_server_url(llm_args: "TorchLlmArgs") -> str:
    """Resolve the server URL: connector-config field > env var > default."""
    cfg = llm_args.kv_connector_config
    if cfg is not None and cfg.server_url is not None:
        return cfg.server_url
    return os.environ.get("LMCACHE_SERVER_URL", DEFAULT_SERVER_URL)


def _send_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list,
) -> MessagingFuture:
    return mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )


def _completed_future(result: bool) -> MessagingFuture:
    """Return an already-completed future resolving to ``result``."""
    future: MessagingFuture = MessagingFuture()
    future.set_result(result)
    return future


@dataclass
class _BlockSpec:
    tokens: List[int]
    block_ids: List[int]


@dataclass
class LMCacheMPConnectorMetadata:
    loads: dict = field(default_factory=dict)
    saves: dict = field(default_factory=dict)


class LMCacheMPKvConnectorScheduler(KvCacheConnectorScheduler):
    """TRT-LLM scheduler that routes lookup requests to an LMCache MP server.

    Returns ``is_async=True`` from :meth:`get_num_new_matched_tokens` so
    that TRT-LLM parks the request in a loading state while the KV
    transfer completes asynchronously.
    """

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size: int = self._llm_args.kv_cache_config.tokens_per_block
        # request_id -> (all_tokens, num_matched).
        self._pending: dict = {}
        # request_ids with in-flight saves — request_finished returns True for these.
        self._saving_in_flight: Set[int] = set()

        self._zmq_context = zmq.Context()
        self._mq_client = MessageQueueClient(
            _get_server_url(self._llm_args), self._zmq_context
        )
        self._mq_timeout = float(
            os.environ.get("LMCACHE_MQ_TIMEOUT", DEFAULT_MQ_TIMEOUT)
        )

        future = _send_request(self._mq_client, RequestType.GET_CHUNK_SIZE, [])
        self._chunk_size = future.result(timeout=self._mq_timeout)
        logger.info(
            "LMCache MP scheduler: connected to server at %s (chunk_size=%d)",
            _get_server_url(self._llm_args),
            self._chunk_size,
        )

        # Third Party
        import tensorrt_llm

        self._rank = tensorrt_llm.mpi_rank()
        tp_size = llm_args.tensor_parallel_size
        pp_size = llm_args.pipeline_parallel_size
        self._world_size = tp_size * pp_size
        self._model_name = str(getattr(llm_args, "model", "unknown_model"))

    def _create_key(
        self,
        token_ids: List[int],
        start: int,
        end: int,
        request_id: int,
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=str(request_id),
        )

    def get_num_new_matched_tokens(
        self,
        request: LlmRequest,
        num_computed_tokens: int,
    ) -> Tuple[int, bool]:
        """Return how many additional tokens the LMCache server can provide.

        Returns ``(new_matched, is_async=True)`` when there are tokens to
        load. TRT-LLM's runtime will park the request in
        ``DISAGG_GENERATION_TRANS_IN_PROGRESS`` state, and the worker's
        :meth:`get_finished` will report it when loading completes.

        Args:
            request: The incoming request with its full token sequence.
            num_computed_tokens: Tokens already matched on device
                (block-aligned).

        Returns:
            ``(new_matched, is_async)``.
        """
        t0 = time.perf_counter()

        if num_computed_tokens % self._block_size != 0:
            self._pending[request.request_id] = ([], 0)
            return 0, False

        all_tokens = list(request.get_tokens(0))

        max_block_aligned = (len(all_tokens) // self._block_size) * self._block_size
        if num_computed_tokens >= max_block_aligned:
            self._pending[request.request_id] = (all_tokens, 0)
            return 0, False

        aligned_end = (len(all_tokens) // self._chunk_size) * self._chunk_size
        key = self._create_key(
            all_tokens, start=0, end=aligned_end, request_id=request.request_id
        ).no_worker_id_version()

        t1 = time.perf_counter()

        try:
            _send_request(self._mq_client, RequestType.LOOKUP, [key, 1]).result(
                timeout=self._mq_timeout
            )
            result = _send_request(
                self._mq_client,
                RequestType.QUERY_PREFETCH_STATUS,
                [str(request.request_id)],
            ).result(timeout=self._mq_timeout)
            cached_tokens = result * self._chunk_size if result is not None else 0
        except Exception as e:
            logger.warning("LMCache MP scheduler: lookup failed: %s", e)
            self._pending[request.request_id] = (all_tokens, 0)
            return 0, False

        t2 = time.perf_counter()

        new_matched = max(0, cached_tokens - num_computed_tokens)
        new_matched = (new_matched // self._block_size) * self._block_size

        # Release read locks on chunks already held by TRT-LLM (overlap).
        overlap_end = min(cached_tokens, num_computed_tokens)
        overlap_end = (overlap_end // self._chunk_size) * self._chunk_size
        if overlap_end > 0:
            free_key = self._create_key(
                all_tokens,
                start=0,
                end=overlap_end,
                request_id=request.request_id,
            ).no_worker_id_version()
            try:
                _send_request(
                    self._mq_client,
                    RequestType.FREE_LOOKUP_LOCKS,
                    [free_key, 1],
                )
            except Exception as e:
                logger.warning("LMCache MP scheduler: free_lookup_locks failed: %s", e)

        self._pending[request.request_id] = (all_tokens, new_matched)

        # Return is_async=True when there are tokens to load. TRT-LLM will
        # park the request and rely on get_finished to report load completion.
        is_async = new_matched > 0

        logger.debug(
            "LMCache MP scheduler: req %d lookup=%.3fms total=%.3fms "
            "trt_matched=%d lmcache_cached=%d new_matched=%d is_async=%s",
            request.request_id,
            (t2 - t1) * 1000,
            (time.perf_counter() - t0) * 1000,
            num_computed_tokens,
            cached_tokens,
            new_matched,
            is_async,
        )
        return new_matched, is_async

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> LMCacheMPConnectorMetadata:
        """Build per-request load/save specs from pending lookup results.

        Requests that appear in ``meta.saves`` are pre-registered in
        ``_saving_in_flight`` so that :meth:`request_finished` can
        immediately return ``True`` when TRT-LLM asks — the worker's
        STORE hasn't fired yet at this point, but it will in the same step.
        """
        meta = LMCacheMPConnectorMetadata()

        for req in scheduler_output.new_requests:
            if req.request_id not in self._pending:
                continue

            all_tokens, num_matched = self._pending[req.request_id]
            block_ids: List[int] = list(req.new_block_ids)
            num_computed_blocks = req.computed_position // self._block_size

            if num_matched > 0:
                meta.loads[req.request_id] = _BlockSpec(
                    tokens=all_tokens, block_ids=block_ids
                )

            save_start = max(num_computed_blocks, num_matched // self._block_size)
            num_full_new_blocks = len(req.new_tokens) // self._block_size
            if (
                save_start < len(block_ids)
                and num_full_new_blocks > 0
                and save_start < num_computed_blocks + num_full_new_blocks
            ):
                meta.saves[req.request_id] = _BlockSpec(
                    tokens=all_tokens, block_ids=block_ids
                )
                # Pre-register: the worker will fire STORE for this request
                # in wait_for_save (same step). request_finished may be
                # called any time after generation ends — this ensures it
                # returns True.
                self._saving_in_flight.add(req.request_id)

        self._pending.clear()
        return meta

    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        """Return whether async saving is in progress.

        Returns ``True`` when a STORE is still in flight for this request.
        TRT-LLM will defer block deallocation until ``get_finished`` on
        the worker reports the request complete.

        The ID is removed from ``_saving_in_flight`` on consumption since
        ``request_finished`` is called exactly once per request by the
        TRT-LLM runtime — keeping it would leak memory over long runs.
        """
        try:
            self._saving_in_flight.remove(request.request_id)
            return True
        except KeyError:
            return False

    def update_state_after_alloc(
        self, request: LlmRequest, block_ids: List[int]
    ) -> None:
        """No-op — block IDs are captured in :meth:`build_connector_meta`."""
        pass


class LMCacheMPKvConnectorWorker(KvCacheConnectorWorker):
    """TRT-LLM worker that routes store/retrieve to an LMCache MP server.

    All operations are non-blocking: futures are held and polled via
    :meth:`get_finished`.
    """

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size: int = self._llm_args.kv_cache_config.tokens_per_block

        self._zmq_context = zmq.Context()
        self._mq_client = MessageQueueClient(
            _get_server_url(self._llm_args), self._zmq_context
        )
        self._mq_timeout = float(
            os.environ.get("LMCACHE_MQ_TIMEOUT", DEFAULT_MQ_TIMEOUT)
        )

        self._instance_id = os.getpid()
        self._registered = False

        # Third Party
        import tensorrt_llm

        self._rank = tensorrt_llm.mpi_rank()
        tp_size = llm_args.tensor_parallel_size
        pp_size = llm_args.pipeline_parallel_size
        self._world_size = tp_size * pp_size
        self._model_name = str(getattr(llm_args, "model", "unknown_model"))

        future = _send_request(self._mq_client, RequestType.GET_CHUNK_SIZE, [])
        self._chunk_size = future.result(timeout=self._mq_timeout)

        # In-flight tracking for async operations.
        # Maps request_id -> (future, export_event_keepalive).
        self._inflight_saves: Dict[int, Tuple[MessagingFuture, object]] = {}
        self._inflight_loads: Dict[int, Tuple[MessagingFuture, object]] = {}

    def _create_key(
        self,
        token_ids: List[int],
        request_id: int,
    ) -> IPCCacheServerKey:
        aligned_end = (len(token_ids) // self._chunk_size) * self._chunk_size
        return IPCCacheServerKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._rank,
            token_ids=tuple(token_ids),
            start=0,
            end=aligned_end,
            request_id=str(request_id),
        )

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor) -> None:
        """Register the KV pool with the LMCache server via raw CUDA IPC.

        TRT-LLM provides a 4-D pool tensor
        ``[NB, NL, 2, NH * BS * HS]``. The server reshapes it to 6-D
        ``[NB, NL, 2, NH, BS, HS]`` from the ``layout_hints`` so format
        detection lands on ``NB_NL_TWO_NH_BS_HS``.
        """
        if self._registered:
            logger.info("LMCache MP worker: KV caches already registered")
            return

        # Third Party
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(self._model_name)
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        num_kv_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        )
        tp_size = self._llm_args.tensor_parallel_size
        num_kv_heads = num_kv_heads // tp_size

        _, _, _, block_size_flat = kv_cache_tensor.shape
        tokens_per_block = block_size_flat // (num_kv_heads * head_dim)

        wrapped = [RawCudaIPCWrapper(kv_cache_tensor)]

        layout_hints = {
            "kv_layout": "HND",
            "num_kv_heads": num_kv_heads,
            "tokens_per_block": tokens_per_block,
            "head_dim": head_dim,
        }

        future = _send_request(
            self._mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                self._instance_id,
                wrapped,
                self._model_name,
                self._world_size,
                EngineType.TRTLLM,
                layout_hints,
                [],
            ],
        )
        try:
            future.result(timeout=self._mq_timeout)
            self._registered = True
            logger.info(
                "LMCache MP worker: registered KV caches "
                "(tensor_shape=%s, NH=%d, BS=%d, HS=%d)",
                list(kv_cache_tensor.shape),
                num_kv_heads,
                tokens_per_block,
                head_dim,
            )
        except TimeoutError:
            logger.error(
                "LMCache MP worker: KV cache registration timed out after %ss",
                self._mq_timeout,
            )

    def start_load_kv(self, stream: torch_dev.Stream) -> None:
        """Submit non-blocking RETRIEVE requests for each pending load.

        Fires RETRIEVE without waiting. The futures are tracked in
        ``_inflight_loads`` and polled by :meth:`get_finished`.
        """
        meta: Optional[LMCacheMPConnectorMetadata] = self._metadata
        if meta is None or not meta.loads:
            return

        t0 = time.perf_counter()
        check_interprocess_event_support()
        event = torch_dev.Event(interprocess=True)
        event.record(stream)

        for req_id, spec in meta.loads.items():
            if not spec.tokens or not spec.block_ids:
                continue

            key = self._create_key(spec.tokens, req_id)
            try:
                future = _send_request(
                    self._mq_client,
                    RequestType.RETRIEVE,
                    [
                        key,
                        self._instance_id,
                        [spec.block_ids],
                        event.ipc_handle(),
                        0,  # skip_first_n_tokens
                    ],
                )
                # Pin the export event to prevent GC before the daemon
                # imports its IPC handle and waits on it.
                self._inflight_loads[req_id] = (future, event)
            except Exception as e:
                logger.warning(
                    "LMCache MP worker: retrieve submit failed for req %d: %s",
                    req_id,
                    e,
                )

        logger.debug(
            "LMCache MP worker: start_load_kv submitted %d loads in %.3fms",
            len(meta.loads),
            (time.perf_counter() - t0) * 1000,
        )

    def wait_for_layer_load(self, layer_idx: int, stream: torch_dev.Stream) -> None:
        """No-op — server synchronizes via CUDA IPC events."""
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch_dev.Stream) -> None:
        """No-op — saves are batched in :meth:`wait_for_save`."""
        pass

    def wait_for_save(self, stream: torch_dev.Stream) -> None:
        """Submit non-blocking STORE requests for each pending save.

        Fires STORE for each request, records a CUDA event on ``stream``
        for the daemon to synchronize against, and keeps the futures for
        polling in :meth:`get_finished`. Does NOT block on ``.result()``.
        """
        meta: Optional[LMCacheMPConnectorMetadata] = self._metadata
        if meta is None or not meta.saves:
            return

        t0 = time.perf_counter()
        check_interprocess_event_support()
        event = torch_dev.Event(interprocess=True)
        event.record(stream)

        for req_id, spec in meta.saves.items():
            if not spec.tokens or not spec.block_ids:
                continue

            key = self._create_key(spec.tokens, req_id)
            try:
                future = _send_request(
                    self._mq_client,
                    RequestType.STORE,
                    [
                        key,
                        self._instance_id,
                        [spec.block_ids],
                        event.ipc_handle(),
                    ],
                )
                # Pin the export event so the daemon's import sees it alive.
                self._inflight_saves[req_id] = (future, event)
            except Exception as e:
                logger.warning(
                    "LMCache MP worker: store submit failed for req %d: %s",
                    req_id,
                    e,
                )

        logger.debug(
            "LMCache MP worker: wait_for_save submitted %d stores in %.3fms",
            len(meta.saves),
            (time.perf_counter() - t0) * 1000,
        )

    def get_finished(
        self,
        finished_gen_req_ids: List[int],
        started_loading_req_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Poll in-flight futures and report completed request IDs.

        Per the TRT-LLM ABC contract (kv_cache_connector.py:174-196):
        - IDs may only be returned from this call after they've been
          provided in ``finished_gen_req_ids`` / ``started_loading_req_ids``.
        - The runtime will only take action once ALL workers report the
          same ID (multi-rank allgather).

        Returns:
            Tuple of (finished_saving_ids, finished_loading_ids).
        """
        finished_saving: List[int] = []
        finished_loading: List[int] = []

        # Poll saves — only IDs that TRT-LLM has acknowledged as
        # "finished generation, now saving" are eligible to report.
        eligible_saves = set(finished_gen_req_ids)
        for req_id in list(self._inflight_saves.keys()):
            if req_id not in eligible_saves:
                continue
            future, _event = self._inflight_saves[req_id]
            if future.query():
                del self._inflight_saves[req_id]
                finished_saving.append(req_id)
                # Fire END_SESSION now that the STORE is complete.
                try:
                    _send_request(
                        self._mq_client,
                        RequestType.END_SESSION,
                        [str(req_id)],
                    )
                except Exception as e:
                    logger.warning(
                        "LMCache MP worker: end_session failed for req %d: %s",
                        req_id, e,
                    )

        # Poll loads — only IDs that TRT-LLM has acknowledged as
        # "started loading" are eligible to report.
        eligible_loads = set(started_loading_req_ids)
        for req_id in list(self._inflight_loads.keys()):
            if req_id not in eligible_loads:
                continue
            future, _event = self._inflight_loads[req_id]
            if future.query():
                del self._inflight_loads[req_id]
                finished_loading.append(req_id)

        if finished_saving or finished_loading:
            logger.debug(
                "LMCache MP worker: get_finished saves=%s loads=%s "
                "(pending_saves=%d pending_loads=%d)",
                finished_saving,
                finished_loading,
                len(self._inflight_saves),
                len(self._inflight_loads),
            )

        return finished_saving, finished_loading
