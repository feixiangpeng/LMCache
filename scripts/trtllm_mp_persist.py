# SPDX-License-Identifier: Apache-2.0
"""E2E persistence test: the LMCache MP server outlives a TRT-LLM restart.

The multi-process cache server holds L1 in its own process, independent
of any TRT-LLM engine. This test proves that KV stored by one engine is
served to a *different* engine created after the first was fully torn
down -- i.e. the cache persists across an engine restart.

Design (why it is unambiguous):

    FILL     new engine #1  -> generate P  -> new_matched == 0  (cold)
    <engine #1 process EXITS -- GPU pool, block reuse, executor gone>
    REVISIT  new engine #2  -> generate P  -> new_matched  > 0  (warm)

The two engines run in *separate OS processes*, each spawned by the
driver, against the *same* long-lived MP server (started by
trtllm_mp_persist.sh). When engine #1's process exits, TRT-LLM's own KV
pool and in-process block reuse die with it, so engine #2 starts with an
empty GPU cache. Any hit engine #2 sees therefore cannot come from
TRT-LLM reuse -- it can only come from the persistent MP server. No
eviction fillers are needed: a brand-new engine has nothing cached
locally. This is a stronger isolation of the "server persists" property
than the cold/evict/warm pattern, which only evicts within one engine.

Persistence works because ``IPCCacheServerKey`` is keyed by content
(model_name, world_size, worker_id, tokens, cache_salt) -- not by pid or
instance_id. Engine #2 registers a fresh KV pool (new instance_id) but
computes the same key for prompt P, so the lookup hits and the transfer
targets engine #2's newly-registered pool.

Observability is the scheduler's DEBUG lookup line
(``... lmcache_cached=C new_matched=M ...``), same signal as the other
e2e scripts; ``new_matched > 0`` on REVISIT is the persistence proof.

Usage (inside the TRT-LLM release container, 1+ GPU, with a running
LMCache MP server -- see trtllm_mp_persist.sh):
    python3 scripts/trtllm_mp_persist.py --server-url tcp://localhost:5555
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

# One long deterministic prompt (~600 tokens, ~10 blocks). Reused by both
# engines; temperature=0 makes the output byte-identical across the
# store/reload so text equality checks KV correctness.
PROMPT = "The quick brown fox jumps over the lazy dog. " * 60 + " Summarize:"

# Phases, each run in its own engine process.
PHASES = ["FILL", "REVISIT"]
MARKER = "### PERSIST_STEP "
_TEXT_PREFIX = "### PERSIST_TEXT "
_DONE = "### PERSIST_INNER_DONE"

_LOOKUP_RE = re.compile(r"lmcache_cached=(\d+)\s+new_matched=(\d+)")
_MARKER_RE = re.compile(re.escape(MARKER) + r"(\S+)")


def _run_inner(server_url: str, phase: str) -> int:
    """Create ONE fresh TRT-LLM engine, generate PROMPT once, then exit.

    Emits a ``### PERSIST_STEP <phase>`` marker immediately before the
    observed generate so the driver can attribute the scheduler's DEBUG
    lookup line, then prints the generated text. The engine is destroyed
    when this process exits -- which, for the REVISIT phase, is exactly
    what makes the subsequent hit attributable solely to the MP server.
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
        max_batch_size=2,
        max_seq_len=2048,
    )

    params = SamplingParams(max_tokens=16, temperature=0.0)

    print(f"\n{MARKER}{phase}", flush=True)
    text = llm.generate([PROMPT], params)[0].outputs[0].text
    print(f"{_TEXT_PREFIX}{phase} {text!r}", flush=True)

    print(_DONE, flush=True)
    return 0


def _parse(log: str, phase: str) -> tuple:
    """Extract the lookup and text for ``phase`` from one inner run's log.

    Returns:
        ``(lookup, text)`` where ``lookup`` is the first
        ``{"cached", "new_matched"}`` seen after the phase marker (or
        ``None`` if the request short-circuited), and ``text`` is the
        generated text (or ``None`` if absent).
    """
    lookup = None
    text = None
    current = ""
    for line in log.splitlines():
        m = _MARKER_RE.search(line)
        if m:
            current = m.group(1)
            continue
        idx = line.find(_TEXT_PREFIX)
        if idx != -1:
            label, _, text_repr = line[idx + len(_TEXT_PREFIX):].partition(" ")
            if label == phase:
                text = text_repr
            continue
        if current == phase and lookup is None:
            lm = _LOOKUP_RE.search(line)
            if lm:
                lookup = {
                    "cached": int(lm.group(1)),
                    "new_matched": int(lm.group(2)),
                }
    return lookup, text


def _spawn_engine(server_url: str, phase: str) -> tuple:
    """Spawn one inner engine process for ``phase`` and capture its log.

    Returns:
        ``(returncode, log)``.
    """
    env = dict(os.environ)
    env["LMCACHE_LOG_LEVEL"] = "DEBUG"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), "--inner",
           "--phase", phase, "--server-url", server_url]
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="tcp://localhost:5555")
    parser.add_argument("--inner", action="store_true",
                        help="Internal: run one engine + generate (child role).")
    parser.add_argument("--phase", choices=PHASES,
                        help="Internal: which phase this child runs.")
    args = parser.parse_args()

    config_path = os.path.join(tempfile.gettempdir(),
                               "lmcache_persist_config.yaml")
    with open(config_path, "w") as f:
        f.write(CONFIG_YAML)
    os.environ["LMCACHE_CONFIG_FILE"] = config_path

    if args.inner:
        return _run_inner(args.server_url, args.phase or "FILL")

    print(f"=== MP persistence e2e (server={args.server_url}) ===", flush=True)

    results = {}
    for phase in PHASES:
        print(f"\n----- engine process: {phase} -----", flush=True)
        rc, log = _spawn_engine(args.server_url, phase)
        sys.stdout.write(log)
        sys.stdout.flush()
        if rc != 0:
            print(f"PERSIST TEST FAILED: {phase} engine exited {rc}", flush=True)
            return 1
        if _DONE not in log:
            print(f"PERSIST TEST FAILED: {phase} engine did not complete",
                  flush=True)
            return 1
        results[phase] = _parse(log, phase)

    print("=" * 60, flush=True)
    lookups = {p: results[p][0] for p in PHASES}
    print(f"observed lookups: {lookups}", flush=True)

    failures = []

    def nm(phase: str) -> int:
        lk = results[phase][0]
        return lk["new_matched"] if lk else -1

    for phase in PHASES:
        if results[phase][0] is None:
            failures.append(
                f"{phase}: no scheduler lookup observed (short-circuit?)"
            )

    # FILL is the first request against a fresh cache -> cold.
    if results["FILL"][0] is not None and nm("FILL") != 0:
        failures.append(f"FILL expected new_matched=0 (cold), got {nm('FILL')}")

    # REVISIT runs in a brand-new engine (empty GPU pool) -> any hit is
    # from the persistent MP server, proving cache survived the restart.
    if results["REVISIT"][0] is not None and nm("REVISIT") <= 0:
        failures.append(
            f"REVISIT expected new_matched>0 (cache must survive engine "
            f"restart via the persistent MP server), got {nm('REVISIT')}"
        )

    # KV correctness across the restart round-trip.
    fill_text, revisit_text = results["FILL"][1], results["REVISIT"][1]
    if fill_text is not None and revisit_text is not None and \
            fill_text != revisit_text:
        failures.append(
            f"REVISIT text {revisit_text} != FILL {fill_text} "
            f"(persisted KV may be corrupt)"
        )

    if failures:
        print(f"PERSIST TEST FAILED: {len(failures)} problem(s)", flush=True)
        for f_ in failures:
            print(f"  - {f_}", flush=True)
        return 1

    print("PERSIST TEST PASSED: MP server served a warm hit to a fresh "
          "engine after the original was destroyed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
