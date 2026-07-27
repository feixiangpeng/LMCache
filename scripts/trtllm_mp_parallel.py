# SPDX-License-Identifier: Apache-2.0
"""E2E test for the MP TRT-LLM LMCache connector under model parallelism.

Generalizes the TP=2 e2e (trtllm_mp_tp2.py) to arbitrary tensor-parallel
(``--tp``) and pipeline-parallel (``--pp``) degrees. The coordination
paths under test are the same, just at higher rank counts:

  * Per-rank worker shards. Each rank runs its own
    ``LMCacheMPKvConnectorWorker`` (``mpi_rank()`` -> 0..world_size-1),
    keys its KV shard under ``worker_id=self._rank`` and registers a
    pool shard with its own ``instance_id``. Under TP the pool is split
    across heads; under PP each rank owns a disjoint slice of layers.
    All ranks must store/retrieve their shards independently.
  * The CUDA-IPC import must open each rank's exported KV pool on that
    rank's own physical device. This is exactly the TP>1 wrong-device
    bug fixed in ipc_wrapper.py: at TP=2 only rank 1 -> cuda:1 is a
    non-default mapping, so TP>=4 (cuda:1/2/3...) is a stronger test
    that the fix generalizes beyond the N=2 case.
  * Multi-rank allgather in ``get_finished``. TRT-LLM only resumes a
    parked async-load request once *every* rank reports it finished
    loading. Loads have no immediate-report fallback, so a stalled or
    missing shard on ANY rank stalls the request (visible hang) rather
    than resuming against partially-loaded KV (silent corruption). A
    correct warm revisit with byte-identical output proves every rank's
    shard was stored and reloaded and the allgather released the request.

Observability: the connector scheduler is a singleton (runs in the
coordinator), so it emits ONE ``... lmcache_cached=.. new_matched=..``
DEBUG line per request regardless of parallelism degree —
``new_matched > 0`` on the warm revisit is the LMCache-hit signal. The
per-rank workers additionally emit registration lines; how many surface
depends on whether every rank's stderr funnels into the captured pipe
(rank 0's does; others are best-effort under MPI), so their count is
*reported*, not asserted. The asserted correctness gate is: warm
``new_matched > 0`` AND cold-text == warm-text.

Step sequence (one LLM instance, GPU pool sized so the revisit is
evicted from TRT-LLM's own reuse -> any warm hit is from LMCache):

    COLD   prompt P  -> new_matched == 0   (nothing cached yet)
    <filler>         -> evict P from every rank's GPU pool
    WARM   prompt P  -> new_matched  > 0   (all shards reloaded)

Two hard constraints discovered while validating higher degrees (both
enforced upstream, before the connector runs -- they are model/engine
limits, not LMCache limits):

  1. KV-head divisibility. TP=N requires ``num_kv_heads % N == 0``
     (TRT-LLM asserts ``num_heads % (tp_size * cp_size) == 0``).
     Qwen2.5-0.5B-Instruct has only 2 KV heads (GQA) -> TP<=2 only; use
     a model with >=N KV heads for TP=N (e.g. TinyLlama-1.1B has 4, so
     ``SMOKE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0`` for TP=4).
  2. ``--pp`` (and context parallelism) is UNSUPPORTED with any KV
     connector: TRT-LLM raises ``NotImplementedError: KV Cache
     Connector is not supported with pipeline or context parallelism``
     because a connector worker registers only its local rank's pool
     with no cross-rank coordination. ``--pp`` is kept here so the
     rejection is reproducible, but only ``--pp 1`` can pass.

Note on the byte-identical gate: exact-token reproducibility across the
store/reload round-trip is a *model* property, not a connector one.
Qwen2.5-0.5B is bit-exact (COLD text == WARM text). TinyLlama-1.1B is
not -- it drifts to a different (still coherent) continuation, and it
does so identically at TP=1, TP=2, and TP=4, proving the drift is
model-level numerical sensitivity and NOT a sharding/IPC/allgather bug.
Use a bit-exact model when you want the text-equality assertion to hold;
for a >=4-head model that only drifts numerically, the meaningful TP
assertions are COLD new_matched==0, WARM new_matched>0, and N shards
registered with no CUDA fault.

Usage (inside the TRT-LLM release container, tp*pp GPUs visible, with a
running LMCache MP server -- see trtllm_mp_parallel.sh):
    python3 scripts/trtllm_mp_parallel.py --server-url tcp://localhost:5555 \
        --tp 4 --pp 1
"""

# Standard
import argparse
import os
import re
import subprocess
import sys
import tempfile

CONFIG_YAML = """\
chunk_size: 64
local_cpu: true
max_local_cpu_size: 4.0
"""

# One long deterministic prompt (~600 tokens, ~10 blocks) reused across
# the cold fill and the warm revisit.
PROMPT = "The quick brown fox jumps over the lazy dog. " * 60 + " Summarize:"

# Two distinct fillers pushed between cold and warm to flush PROMPT out
# of every rank's GPU block reuse; sized (each ~550-650 tokens) so
# PROMPT + both fillers cannot coexist in the 1280-token pool but no
# single request overflows it.
FILLERS = [
    "Alpha series: " + " ".join(str(i) for i in range(150)),
    "Beta series: " + " ".join(str(i) for i in range(600, 750)),
]

STEP_ORDER = ["COLD", "WARM"]
MARKER = "### PAR_STEP "
_TEXT_PREFIX = "### PAR_TEXT "
_DONE = "### PAR_INNER_DONE"

# Matches the MP scheduler's per-request lookup line.
_LOOKUP_RE = re.compile(r"lmcache_cached=(\d+)\s+new_matched=(\d+)")
_MARKER_RE = re.compile(re.escape(MARKER) + r"(\S+)")
# Worker registration line (one per rank whose stderr is captured).
_REGISTER_RE = re.compile(r"registered KV caches \(tensor_shape=(\[[^\]]*\])")


def _run_inner(server_url: str, tp: int, pp: int) -> int:
    """Run the cold/evict/warm sequence on a TP*PP LLM (child role).

    Emits a ``### PAR_STEP <label>`` marker immediately before each
    observed generate so the driver can attribute the scheduler's DEBUG
    lookup line to the right step. Returns 0 once the sequence completes;
    correctness is asserted by the driver from the captured log.

    Args:
        server_url: ZMQ URL of the running LMCache MP cache server.
        tp: Tensor-parallel degree passed to ``tensor_parallel_size``.
        pp: Pipeline-parallel degree passed to ``pipeline_parallel_size``.

    Returns:
        Process exit code (0 on completion).
    """
    # Third Party
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import (
        KvCacheConfig,
        KvCacheConnectorConfig,
    )

    connector_cfg = KvCacheConnectorConfig(
        connector="lmcache-mp", server_url=server_url
    )
    kv_cfg = KvCacheConfig(
        max_tokens=1280,
        tokens_per_block=64,
        enable_block_reuse=True,
    )

    llm = LLM(
        model=os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        kv_cache_config=kv_cfg,
        kv_connector_config=connector_cfg,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        max_batch_size=2,
        max_seq_len=2048,
    )

    params = SamplingParams(max_tokens=16, temperature=0.0)

    def observed(label: str) -> str:
        print(f"\n{MARKER}{label}", flush=True)
        text = llm.generate([PROMPT], params)[0].outputs[0].text
        print(f"{_TEXT_PREFIX}{label} {text!r}", flush=True)
        return text

    def evict() -> None:
        print(f"\n{MARKER}FILLER", flush=True)
        for filler in FILLERS:
            llm.generate([filler + " What comes next?"], params)

    observed("COLD")
    evict()
    observed("WARM")

    print(_DONE, flush=True)
    return 0


def _parse(log: str) -> tuple:
    """Parse per-step lookups, texts, and worker registrations from a log.

    Returns:
        ``(steps, texts, registrations)`` where ``steps`` maps each step
        label to the first ``{"cached", "new_matched"}`` lookup after its
        marker, ``texts`` maps each label to its generated text, and
        ``registrations`` is the list of registered pool shard shapes
        (one per rank whose stderr was captured).
    """
    steps: dict = {}
    texts: dict = {}
    registrations: list = []
    current = ""
    for line in log.splitlines():
        m = _MARKER_RE.search(line)
        if m:
            current = m.group(1)
            continue
        idx = line.find(_TEXT_PREFIX)
        if idx != -1:
            label, _, text_repr = line[idx + len(_TEXT_PREFIX):].partition(" ")
            if label in STEP_ORDER:
                texts[label] = text_repr
            continue
        reg = _REGISTER_RE.search(line)
        if reg:
            registrations.append(reg.group(1))
            continue
        if current in STEP_ORDER and current not in steps:
            lm = _LOOKUP_RE.search(line)
            if lm:
                steps[current] = {
                    "cached": int(lm.group(1)),
                    "new_matched": int(lm.group(2)),
                }
    return steps, texts, registrations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="tcp://localhost:5555")
    parser.add_argument("--tp", type=int, default=2,
                        help="Tensor-parallel degree.")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline-parallel degree.")
    parser.add_argument("--allow-text-drift", action="store_true",
                        help="Downgrade a COLD!=WARM text mismatch from a "
                             "failure to a warning. Use for models that are "
                             "not bit-exact across the reload round-trip "
                             "(e.g. TinyLlama); the warm-hit gate still holds.")
    parser.add_argument("--inner", action="store_true",
                        help="Internal: run the generate sequence (child role).")
    args = parser.parse_args()

    config_path = os.path.join(tempfile.gettempdir(),
                               "lmcache_parallel_config.yaml")
    with open(config_path, "w") as f:
        f.write(CONFIG_YAML)
    os.environ["LMCACHE_CONFIG_FILE"] = config_path

    if args.inner:
        return _run_inner(args.server_url, args.tp, args.pp)

    # Driver role: spawn the inner run, capture combined output, tee it,
    # then parse and assert.
    env = dict(os.environ)
    env["LMCACHE_LOG_LEVEL"] = "DEBUG"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, os.path.abspath(__file__), "--inner",
           "--server-url", args.server_url,
           "--tp", str(args.tp), "--pp", str(args.pp)]

    world = args.tp * args.pp
    print(f"=== TP={args.tp} PP={args.pp} (world={world}) MP e2e "
          f"(server={args.server_url}) ===", flush=True)
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    log = proc.stdout
    sys.stdout.write(log)
    sys.stdout.flush()

    if proc.returncode != 0:
        print(f"PARALLEL TEST FAILED: inner run exited {proc.returncode}",
              flush=True)
        return 1
    if _DONE not in log:
        print("PARALLEL TEST FAILED: inner run did not complete", flush=True)
        return 1

    steps, texts, registrations = _parse(log)
    print("=" * 60, flush=True)
    print(f"observed lookups: {steps}", flush=True)
    print(f"worker registrations captured: {len(registrations)} "
          f"(shards={registrations}; 1 per rank whose stderr surfaced)",
          flush=True)

    failures = []

    def nm(label: str) -> int:
        return steps.get(label, {}).get("new_matched", -1)

    for label in STEP_ORDER:
        if label not in steps:
            failures.append(
                f"{label}: no scheduler lookup observed "
                f"(short-circuit? eviction may have failed)"
            )

    # Cold: nothing cached yet.
    if "COLD" in steps and nm("COLD") != 0:
        failures.append(f"COLD expected new_matched=0, got {nm('COLD')}")

    # Warm: every rank shard must have been stored then reloaded. A
    # multi-rank allgather stall or a missing shard would leave this 0.
    if "WARM" in steps and nm("WARM") <= 0:
        failures.append(
            f"WARM expected new_matched>0 (all {world} rank shards reloaded "
            f"via multi-rank allgather), got {nm('WARM')}"
        )

    # KV correctness across the sharded round-trip. Byte-identical output
    # is only expected from bit-exact models; a numerically-sensitive
    # model (e.g. TinyLlama) drifts to a different but coherent
    # continuation even at TP=1 (no sharding), so --allow-text-drift
    # downgrades the mismatch to a warning while keeping the warm-hit gate.
    if "COLD" in texts and "WARM" in texts and texts["COLD"] != texts["WARM"]:
        msg = (
            f"WARM text {texts['WARM']} != COLD {texts['COLD']} "
            f"(bit-exact reload not observed for this model)"
        )
        if args.allow_text_drift:
            print(f"WARNING: {msg}; treating as text drift, not a failure "
                  f"(--allow-text-drift set)", flush=True)
        else:
            failures.append(msg + " (sharded async load may have corrupted KV)")

    if failures:
        print(f"PARALLEL TEST FAILED: {len(failures)} problem(s)", flush=True)
        for f_ in failures:
            print(f"  - {f_}", flush=True)
        return 1

    text_note = ("byte-identical output"
                 if texts.get("COLD") == texts.get("WARM")
                 else "coherent output (text drift allowed)")
    print(f"PARALLEL TEST PASSED: TP={args.tp} PP={args.pp} warm hit + "
          f"{text_note} (all {world} rank shards stored and reloaded)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
