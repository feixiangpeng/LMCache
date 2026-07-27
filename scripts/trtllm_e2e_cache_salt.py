# SPDX-License-Identifier: Apache-2.0
"""E2E cache-salt isolation test for the async TRT-LLM LMCache connector.

`cache_salt` partitions the cache: two requests with identical tokens but
different salts must NOT share KV (multi-tenant isolation), while two
requests with identical tokens AND the same salt MUST share.

Under ``temperature=0`` the generated text is byte-identical whether KV
came from cache or was recomputed, so output equality cannot prove
isolation. The ground-truth signal is the scheduler's DEBUG log line:

    LMCache TRT-LLM scheduler: req N ... lmcache_cached=C new_matched=M ...
    LMCache MP scheduler:      req N ... lmcache_cached=C new_matched=M ...

``new_matched > 0`` means LMCache served a hit for that request. The
connector scheduler runs in TRT-LLM's child executor process, so we
cannot observe the signal in-process. Instead the driver runs the
generate sequence in a subprocess, captures its full combined output
(the child's stderr is inherited into the pipe), and parses the
``new_matched`` value attributed to each labelled step.

Step sequence (one LLM instance, GPU pool sized so each revisit is
evicted from TRT-LLM's own reuse and any hit must come from LMCache):

    A_FIRST   prompt P, salt "tenant-a"  -> cold  (new_matched == 0)
    <filler>                             -> evict P from GPU pool
    B_FIRST   prompt P, salt "tenant-b"  -> MISS  (new_matched == 0)   [isolation]
    <filler>                             -> evict
    A_SECOND  prompt P, salt "tenant-a"  -> HIT   (new_matched  > 0)   [same salt shares]
    <filler>                             -> evict
    B_SECOND  prompt P, salt "tenant-b"  -> HIT   (new_matched  > 0)   [tenant-b namespace works]

The B_FIRST==0 assertion is the isolation proof: despite A_FIRST having
filled the cache for prompt P, tenant-b must see nothing. A_SECOND>0 is
the positive control and doubles as a guard that eviction really
happened (if the pool never evicted, A_SECOND would short-circuit to 0
and fail loudly rather than pass silently).

Usage (inside the TRT-LLM release container with lmcache installed):
    python3 scripts/trtllm_e2e_cache_salt.py --preset lmcache
    python3 scripts/trtllm_e2e_cache_salt.py --preset lmcache-mp \\
        --server-url tcp://localhost:5555
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

# Salt values. Both are safe under IPCCacheServerKey validation
# (<=128 chars, no @ / \\ NUL).
SALT_A = "tenant-a"
SALT_B = "tenant-b"

# One long deterministic prompt (~600 tokens, ~10 blocks) reused across
# every logical request.
PROMPT = "The quick brown fox jumps over the lazy dog. " * 60 + " Summarize:"

# Two distinct filler prompts pushed between steps to flush PROMPT out of
# TRT-LLM's own GPU block reuse, so any subsequent hit can only come from
# LMCache. Sizing constraints, with KvCacheConfig(max_tokens=1280):
#   * Each must fit: input + generated (16) <= max_seq_len, which TRT-LLM
#     caps at the pool size (1280). ~550-650 tokens each stays well under.
#   * Together they must evict PROMPT: PROMPT (~600) + both fillers (~1200)
#     cannot coexist in a 1280-token pool, so the LRU entry (PROMPT) is
#     reclaimed. Two distinct fillers make eviction robust to token-count
#     estimation error (a single tight-margin filler risks either a
#     max_seq_len overflow crash or under-eviction).
FILLERS = [
    "Alpha series: " + " ".join(str(i) for i in range(150)),
    "Beta series: " + " ".join(str(i) for i in range(600, 750)),
]

# Ordered labels for the four observed requests (fillers are unlabelled;
# their lookups are ignored by the parser).
STEP_ORDER = ["A_FIRST", "B_FIRST", "A_SECOND", "B_SECOND"]

# Emitted by the inner run right before each observed generate() call.
MARKER = "### SALT_STEP "

# Matches both in-process and MP scheduler lookup lines.
_LOOKUP_RE = re.compile(r"lmcache_cached=(\d+)\s+new_matched=(\d+)")
_MARKER_RE = re.compile(re.escape(MARKER) + r"(\S+)")


def _run_inner(preset: str, server_url: str) -> int:
    """Execute the generate sequence in-process (child role).

    Prints a ``### SALT_STEP <label>`` marker to stdout immediately
    before each observed request so the driver can attribute the
    scheduler's DEBUG lookup line to the right logical step. Returns 0
    once the sequence completes; correctness assertions are made by the
    driver from the captured log.
    """
    # Third Party
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import (
        KvCacheConfig,
        KvCacheConnectorConfig,
    )

    connector_kwargs = {"connector": preset}
    if server_url:
        connector_kwargs["server_url"] = server_url
    connector_cfg = KvCacheConnectorConfig(**connector_kwargs)

    # Pool holds ~1280 tokens (~one prompt). The ~900-token filler
    # guarantees prompt P is evicted from TRT-LLM's own reuse between
    # steps, so a later LMCache hit is unambiguous.
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

    def observed(label: str, salt: str) -> str:
        # The marker must land in the log strictly before the request's
        # scheduler lookup line. flush=True + PYTHONUNBUFFERED keep
        # stdout ordered against the child's stderr logging.
        print(f"\n{MARKER}{label}", flush=True)
        out = llm.generate([PROMPT], params, cache_salt=salt)
        text = out[0].outputs[0].text
        print(f"### SALT_TEXT {label} {text!r}", flush=True)
        return text

    def evict() -> None:
        print("\n### SALT_STEP FILLER", flush=True)
        for filler in FILLERS:
            llm.generate([filler + " What comes next?"], params)

    observed("A_FIRST", SALT_A)
    evict()
    observed("B_FIRST", SALT_B)
    evict()
    observed("A_SECOND", SALT_A)
    evict()
    observed("B_SECOND", SALT_B)

    print("### SALT_INNER_DONE", flush=True)
    return 0


def _parse_steps(log: str) -> dict:
    """Attribute each scheduler lookup line to the most recent step marker.

    Walks the captured log in order; the ``new_matched`` value from the
    first lookup line seen after a ``### SALT_STEP <label>`` marker is
    recorded for that label. Filler steps (label ``FILLER``) are ignored.

    Args:
        log: The full combined stdout+stderr of the inner run.

    Returns:
        Mapping from step label to ``{"cached": int, "new_matched": int}``.
        Labels with no observed lookup line are absent.
    """
    steps: dict = {}
    current = ""
    for line in log.splitlines():
        m = _MARKER_RE.search(line)
        if m:
            current = m.group(1)
            continue
        if current in STEP_ORDER and current not in steps:
            lm = _LOOKUP_RE.search(line)
            if lm:
                steps[current] = {
                    "cached": int(lm.group(1)),
                    "new_matched": int(lm.group(2)),
                }
    return steps


def _parse_texts(log: str) -> dict:
    """Extract the generated text emitted for each observed step.

    Args:
        log: The full combined stdout+stderr of the inner run.

    Returns:
        Mapping from step label to the raw generated text.
    """
    texts: dict = {}
    prefix = "### SALT_TEXT "
    for line in log.splitlines():
        idx = line.find(prefix)
        if idx == -1:
            continue
        rest = line[idx + len(prefix):]
        label, _, text_repr = rest.partition(" ")
        if label in STEP_ORDER:
            texts[label] = text_repr
    return texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="lmcache",
                        choices=["lmcache", "lmcache-mp"])
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--inner", action="store_true",
                        help="Internal: run the generate sequence (child role).")
    args = parser.parse_args()

    config_path = os.path.join(tempfile.gettempdir(), "lmcache_salt_config.yaml")
    with open(config_path, "w") as f:
        f.write(CONFIG_YAML)
    os.environ["LMCACHE_CONFIG_FILE"] = config_path

    if args.inner:
        return _run_inner(args.preset, args.server_url or "")

    # Driver role: spawn the inner run, capture its full combined output
    # (the child executor's stderr is inherited into the pipe), tee it so
    # it appears in the surrounding log, then parse and assert.
    env = dict(os.environ)
    env["LMCACHE_LOG_LEVEL"] = "DEBUG"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, os.path.abspath(__file__), "--inner",
           "--preset", args.preset]
    if args.server_url:
        cmd += ["--server-url", args.server_url]

    print(f"=== cache-salt isolation e2e (preset={args.preset}) ===", flush=True)
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    log = proc.stdout
    # Tee the inner output so it is preserved in the driver's own log.
    sys.stdout.write(log)
    sys.stdout.flush()

    if proc.returncode != 0:
        print(f"CACHE-SALT TEST FAILED: inner run exited {proc.returncode}",
              flush=True)
        return 1
    if "### SALT_INNER_DONE" not in log:
        print("CACHE-SALT TEST FAILED: inner run did not complete", flush=True)
        return 1

    steps = _parse_steps(log)
    texts = _parse_texts(log)
    print("=" * 60, flush=True)
    print(f"observed lookups: {steps}", flush=True)

    failures = []

    # Every observed step must have produced a lookup line. A missing
    # label means the request short-circuited (TRT-LLM already had all
    # blocks) — for A_SECOND/B_SECOND that would mean eviction failed and
    # the test is not actually exercising LMCache; treat as failure.
    for label in STEP_ORDER:
        if label not in steps:
            failures.append(
                f"{label}: no scheduler lookup observed "
                f"(short-circuit? eviction may have failed)"
            )

    def nm(label: str) -> int:
        return steps.get(label, {}).get("new_matched", -1)

    # Cold: nothing cached for tenant-a yet.
    if "A_FIRST" in steps and nm("A_FIRST") != 0:
        failures.append(
            f"A_FIRST expected new_matched=0 (cold), got {nm('A_FIRST')}"
        )

    # ISOLATION: tenant-b must NOT see tenant-a's cache for identical tokens.
    if "B_FIRST" in steps and nm("B_FIRST") != 0:
        failures.append(
            f"B_FIRST expected new_matched=0 (isolation: tenant-b must not "
            f"reuse tenant-a KV), got {nm('B_FIRST')} — CACHE SALT ISOLATION "
            f"BROKEN"
        )

    # SAME-SALT SHARING (+ eviction guard): tenant-a revisit must hit.
    if "A_SECOND" in steps and nm("A_SECOND") <= 0:
        failures.append(
            f"A_SECOND expected new_matched>0 (same salt must share; also "
            f"guards that eviction occurred), got {nm('A_SECOND')}"
        )

    # tenant-b now has its own entry from B_FIRST -> must hit on revisit.
    if "B_SECOND" in steps and nm("B_SECOND") <= 0:
        failures.append(
            f"B_SECOND expected new_matched>0 (tenant-b namespace should "
            f"cache independently), got {nm('B_SECOND')}"
        )

    # KV correctness: cache-served text must equal the originating text.
    if "A_FIRST" in texts and "A_SECOND" in texts and \
            texts["A_FIRST"] != texts["A_SECOND"]:
        failures.append(
            f"A_SECOND text {texts['A_SECOND']} != A_FIRST {texts['A_FIRST']}"
        )
    if "B_FIRST" in texts and "B_SECOND" in texts and \
            texts["B_FIRST"] != texts["B_SECOND"]:
        failures.append(
            f"B_SECOND text {texts['B_SECOND']} != B_FIRST {texts['B_FIRST']}"
        )

    if failures:
        print(f"CACHE-SALT TEST FAILED: {len(failures)} problem(s)", flush=True)
        for f_ in failures:
            print(f"  - {f_}", flush=True)
        return 1

    print("CACHE-SALT TEST PASSED: different salts isolated, same salt shares",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
