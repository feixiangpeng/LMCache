# SPDX-License-Identifier: Apache-2.0
"""E2E test for the MP TRT-LLM LMCache connector under tensor parallelism.

Single-rank MP is covered by the smoke/stress/cache-salt e2e. TP=2
exercises coordination paths those never touch:

  * Per-rank worker shards. Each rank runs its own
    ``LMCacheMPKvConnectorWorker`` (``mpi_rank()`` -> 0 and 1), keys its
    KV shard under ``worker_id=self._rank``, and registers a *sharded*
    pool (``num_kv_heads // tp_size`` heads) with a distinct
    ``instance_id`` (its own pid). Two workers must register two shards
    and store/retrieve them independently.
  * Multi-rank allgather in ``get_finished``. TRT-LLM only resumes a
    parked async-load request once *every* rank reports it finished
    loading. Loads have no immediate-report fallback, so if either
    rank's shard retrieve stalls, the request stalls (visible hang)
    rather than resuming against half-loaded KV (silent corruption).
    A correct warm revisit that returns byte-identical output proves
    both shards were loaded and the allgather released the request.

Observability: the connector scheduler is a singleton (runs in the
coordinator), so it emits ONE ``... trt_matched=.. lmcache_cached=..
new_matched=..`` DEBUG line per request regardless of TP degree —
``new_matched > 0`` on the warm revisit is the LMCache-hit signal, same
as TP=1. The per-rank workers additionally emit registration lines; how
many of those surface depends on whether every rank's stderr funnels
into the captured pipe (rank 0's does; others are best-effort under
MPI), so their count is *reported*, not asserted. The asserted
correctness gate is: warm ``new_matched > 0`` AND cold-text ==
warm-text.

Step sequence (one TP=2 LLM instance, GPU pool sized so the revisit is
evicted from TRT-LLM's own reuse -> any warm hit is from LMCache):

    COLD   prompt P  -> new_matched == 0   (nothing cached yet)
    <filler>         -> evict P from both ranks' GPU pools
    WARM   prompt P  -> new_matched  > 0   (both shards reloaded)

Usage (inside the TRT-LLM release container, 2 GPUs visible, with a
running LMCache MP server — see trtllm_mp_tp2.sh):
    python3 scripts/trtllm_mp_tp2.py --server-url tcp://localhost:5555
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
# of both ranks' GPU block reuse; sized (each ~550-650 tokens) so
# PROMPT + both fillers cannot coexist in the 1280-token pool but no
# single request overflows it. Same rationale as the cache-salt e2e.
FILLERS = [
    "Alpha series: " + " ".join(str(i) for i in range(150)),
    "Beta series: " + " ".join(str(i) for i in range(600, 750)),
]

STEP_ORDER = ["COLD", "WARM"]
MARKER = "### TP2_STEP "
_TEXT_PREFIX = "### TP2_TEXT "
_DONE = "### TP2_INNER_DONE"

# Matches the MP scheduler's per-request lookup line.
_LOOKUP_RE = re.compile(r"lmcache_cached=(\d+)\s+new_matched=(\d+)")
_MARKER_RE = re.compile(re.escape(MARKER) + r"(\S+)")
# Worker registration line (one per rank whose stderr is captured).
_REGISTER_RE = re.compile(r"registered KV caches \(tensor_shape=(\[[^\]]*\])")


def _run_inner(server_url: str) -> int:
    """Run the cold/evict/warm sequence on a TP=2 LLM (child role).

    Emits a ``### TP2_STEP <label>`` marker immediately before each
    observed generate so the driver can attribute the scheduler's DEBUG
    lookup line to the right step. Returns 0 once the sequence completes;
    correctness is asserted by the driver from the captured log.
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
        tensor_parallel_size=2,
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
    parser.add_argument("--inner", action="store_true",
                        help="Internal: run the generate sequence (child role).")
    args = parser.parse_args()

    config_path = os.path.join(tempfile.gettempdir(), "lmcache_tp2_config.yaml")
    with open(config_path, "w") as f:
        f.write(CONFIG_YAML)
    os.environ["LMCACHE_CONFIG_FILE"] = config_path

    if args.inner:
        return _run_inner(args.server_url)

    # Driver role: spawn the inner TP=2 run, capture combined output,
    # tee it, then parse and assert.
    env = dict(os.environ)
    env["LMCACHE_LOG_LEVEL"] = "DEBUG"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, os.path.abspath(__file__), "--inner",
           "--server-url", args.server_url]

    print(f"=== TP=2 MP e2e (server={args.server_url}) ===", flush=True)
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    log = proc.stdout
    sys.stdout.write(log)
    sys.stdout.flush()

    if proc.returncode != 0:
        print(f"TP2 TEST FAILED: inner run exited {proc.returncode}", flush=True)
        return 1
    if _DONE not in log:
        print("TP2 TEST FAILED: inner run did not complete", flush=True)
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

    # Warm: both rank shards must have been stored then reloaded. A
    # multi-rank allgather stall or a missing shard would leave this 0.
    if "WARM" in steps and nm("WARM") <= 0:
        failures.append(
            f"WARM expected new_matched>0 (both TP shards reloaded via "
            f"multi-rank allgather), got {nm('WARM')}"
        )

    # KV correctness across the sharded round-trip.
    if "COLD" in texts and "WARM" in texts and texts["COLD"] != texts["WARM"]:
        failures.append(
            f"WARM text {texts['WARM']} != COLD {texts['COLD']} "
            f"(sharded async load may have corrupted KV)"
        )

    if failures:
        print(f"TP2 TEST FAILED: {len(failures)} problem(s)", flush=True)
        for f_ in failures:
            print(f"  - {f_}", flush=True)
        return 1

    print("TP2 TEST PASSED: TP=2 warm hit + byte-identical output "
          "(both rank shards stored and reloaded)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
