# TensorRT-LLM integration

## Adapter shape

```
lmcache/integration/tensorrt_llm/
├── __init__.py             # Optional-import surface
├── utils.py                # ENGINE_NAME, lmcache_get_config,
│                           # create_trtllm_metadata
├── tensorrt_adapter.py     # In-process — engine in TRT-LLM process
└── tensorrt_mp_adapter.py  # Multi-process — engine in standalone server
```

Both adapters subclass TRT-LLM's `KvCacheConnectorScheduler` and
`KvCacheConnectorWorker`. The TRT-LLM imports are *guarded at module
level only via the package's `__init__`*; nothing in core LMCache
imports the adapter modules. This keeps `pip install lmcache` unaffected
when TRT-LLM is absent.

The TRT-LLM connector preset registry (PR
[NVIDIA/TensorRT-LLM#12626](https://github.com/NVIDIA/TensorRT-LLM/pull/12626))
maps:

| Preset | Module | Scheduler | Worker |
|---|---|---|---|
| `lmcache` | `lmcache.integration.tensorrt_llm.tensorrt_adapter` | `LMCacheKvConnectorScheduler` | `LMCacheKvConnectorWorker` |
| `lmcache-mp` | `lmcache.integration.tensorrt_llm.tensorrt_mp_adapter` | `LMCacheMPKvConnectorScheduler` | `LMCacheMPKvConnectorWorker` |

## Lifecycle

| Stage | TRT-LLM hook | LMCache call |
|---|---|---|
| Init | `worker.register_kv_caches(kv_cache_tensor)` | Build engine via `_get_or_create_engine`; call `gpu_connector.register_kv_caches(kv_cache_tensor)` |
| Before scheduling | `scheduler.get_num_new_matched_tokens(req, num_computed)` | `engine.lookup(tokens)` (in-process) or `LOOKUP` + `QUERY_PREFETCH_STATUS` (MP) |
| Pre-forward | `scheduler.build_connector_meta(scheduler_output)` | `LMCacheConnectorMetadata(loads=..., saves=...)` |
| Forward | `worker.start_load_kv(stream)` | `engine.retrieve(tokens, block_ids)` |
| Forward | `worker.wait_for_save(stream)` | `engine.store(tokens, block_ids)` |

## In-process vs MP

The two modes share the lifecycle but differ in where state lives.

| Aspect | In-process (`lmcache`) | Multi-process (`lmcache-mp`) |
|---|---|---|
| LMCache engine | Singleton inside the TRT-LLM process | Standalone ZMQ server |
| Tensor sharing | Direct (same process) | `RawCudaIPCWrapper` (cudaIpc + cupy DLPack) |
| Lookup | `engine.lookup(tokens)` returns chunk count | `LOOKUP` enqueues prefetch; `QUERY_PREFETCH_STATUS` reads result keyed by `request_id` |
| Configuration | `LMCACHE_CONFIG_FILE` env var | Same; plus `server_url` in connector config (or `LMCACHE_SERVER_URL` env) |
| Failure mode | One process crash takes down both | Engine survives TRT-LLM crash; multiple TRT-LLM instances can share cache |
| Setup cost | None | Run `python -m lmcache.v1.multiprocess.server` |

## Why **not** subclass `VLLMPagedMemGPUConnectorV3`

V3's transfer path is wrong for TRT-LLM:

- **V3 uses the in-process kernel** (`multi_layer_kv_transfer`) with a
  `slot_mapping` of token positions and per-layer pointers. TRT-LLM's
  cross-layer pool is a *single* base pointer and we want to transfer
  by *block ids*, not slot positions.
- **TRT-LLM needs the MP kernel** (`multi_layer_block_kv_transfer`)
  which natively handles single-base-pointer cross-layer with
  `shape_desc.nl` walking layers internally. There is nothing to
  inherit.

`TRTLLMGPUConnector` is therefore a *standalone* `GPUConnectorInterface`
implementation. It also exposes a bespoke
`register_kv_caches(kv_cache_tensor)` method called by the worker once
at init — separate from `to_gpu`/`from_gpu`. The factory in
`lmcache/v1/gpu_connector/__init__.py` constructs it from
`LMCacheMetadata` plus the device, and the adapter wires the pool
tensor in afterwards.

## Async protocol

Both adapters implement fully non-blocking store and load paths using
TRT-LLM's async connector ABC (`get_finished` / `request_finished` /
`is_async` return from `get_num_new_matched_tokens`).

### Store (GPU → cache)

| Step | In-process | Multi-process |
|---|---|---|
| Submit | `engine.store()` enqueued on `store_stream` | `STORE` submitted via ZMQ (fire-and-forget) |
| Track | CUDA event recorded on `store_stream` | `MessagingFuture` + IPC event keepalive |
| Poll | `event.query()` in `get_finished` | `future.query()` in `get_finished` |
| Report | Return req ID in `finished_saving` list | Same |
| Dealloc | `request_finished` → `True` until reported | Same |
| Session cleanup | N/A (in-process) | `END_SESSION` fired only after save completes |

### Load (cache → GPU)

| Step | In-process | Multi-process |
|---|---|---|
| Submit | `engine.retrieve()` on `load_stream` | `RETRIEVE` via ZMQ (non-blocking) |
| Park | `get_num_new_matched_tokens` returns `is_async=True` | Same |
| Track | CUDA event on `load_stream` | `MessagingFuture` + IPC event |
| Report | Return req ID in `finished_loading` list | Same |
| Resume | TRT-LLM reschedules request after all workers report | Same |

### Correctness invariants

1. **Block lifetime**: GPU blocks stay pinned while a save is in flight.
   `request_finished` returns `True`, and TRT-LLM defers `free_resources`
   until `get_finished` reports the ID. Breaking this → silent KV
   corruption from block reuse during active D2H copy.

2. **Eligibility contract**: `get_finished` only returns IDs that have
   been passed in as `finished_gen_req_ids` / `started_loading_req_ids`.
   TRT-LLM provides each ID exactly **once** (it drains
   `new_async_requests` into `pending_async_requests` on the same call),
   so eligibility must be **sticky**: the adapters record granted IDs in
   `_eligible_saves` / `_eligible_loads` and clear them only when
   reported. Treating eligibility as per-call state hangs any operation
   that is still in flight on the call where its ID is first passed.
   The runtime also allgathers across all workers — an ID is actionable
   only once ALL workers have reported it.

3. **No-future fallback (saves only)**: an eligible save ID with no
   in-flight future (submit failed or skipped) is reported immediately —
   otherwise TRT-LLM defers its block deallocation forever. Loads have
   NO such fallback: falsely reporting a load done resumes the request
   against unloaded KV (silent corruption beats a visible stall).

4. **END_SESSION ordering** (MP only): the daemon's `end_session`
   handler cleans up per-request state (lookup locks, session hashes).
   The worker fires `END_SESSION` from `get_finished` only after the
   STORE future completes, ensuring the daemon's STORE handler has
   finished using the session state.

### Async-load scheduling (the parked-request path)

When `get_num_new_matched_tokens` returns `is_async=True`, TRT-LLM
**excludes the request from the scheduler output** (see
`build_scheduler_output` in TRT-LLM's `kv_cache_connector.py`) and
removes it from the scheduled batch. The executor's call order is:

1. `prepare_resources` — allocates blocks (request still in batch),
   calls `update_state_after_alloc(req, block_ids)`, then builds the
   scheduler output **without** the async-loading request.
2. `handle_metadata` — `build_connector_meta(scheduler_output)`; the
   loading request is absent from `scheduler_output.new_requests`.
3. `_kv_connector_start_batch` — removes the request from the batch,
   then calls `start_load_kv`.

The adapters therefore capture block_ids for async-loading requests in
`update_state_after_alloc` (`_pending_async_loads`) and inject them into
`metadata.loads` in `build_connector_meta`. Relying on
`scheduler_output.new_requests` alone deadlocks: the load never starts,
and the request stays parked forever.

### Scheduler ↔ Worker coordination

TRT-LLM constructs the scheduler and worker independently (no shared
constructor argument, no wiring hook). The adapters therefore avoid
cross-references entirely:

- **Scheduler** tracks `_saving_in_flight` on its own: populated at
  `build_connector_meta` time, checked once by `request_finished`.
  Never cleaned up (the runtime only calls `request_finished` once per
  request; the set grows only by concurrent-saves count).
- **Worker** fires `END_SESSION` directly in `get_finished` when a save
  completes — it has its own `_mq_client` to the same daemon, no
  scheduler reference needed.

## Forcing real LMCache hits in tests

TRT-LLM has its own GPU block reuse. To verify LMCache contributes the
hit (and not TRT-LLM's reuse), the E2E tests size TRT-LLM's pool tiny
(`KvCacheConfig(max_tokens=512)`) while sending prompts much larger than
512 tokens. The first request fills LMCache and TRT-LLM. The second is
guaranteed-evicted from TRT-LLM's pool and *must* come from LMCache —
which the test asserts via the `lmcache_cached=… new_matched=…` log
line on request 3.
