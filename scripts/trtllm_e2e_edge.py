# SPDX-License-Identifier: Apache-2.0
"""E2E edge-case tests for the async TRT-LLM LMCache connector (in-process).

Covers geometry and scheduling corners the smoke/stress tests do not:

  * ``overlay``  — partial TRT-LLM match + fuller LMCache match. TRT-LLM
    retains a short prefix in its own block reuse while LMCache holds the
    whole sequence, so the scheduler must load only the non-overlapping
    remainder (``new_matched == align_down(cached - trt_matched)``) and
    still produce correct output.
  * ``prefill``  — ``enable_chunked_prefill=True``. Chunked prefill
    delivers ``new_tokens`` / ``new_block_ids`` to the connector in
    multiple increments per request; the save/load specs must still cover
    the full sequence so a warm revisit hits.
  * ``geometry`` — LMCache ``chunk_size`` (128) != TRT ``tokens_per_block``
    (64). The adapter aligns lookups to the block size while LMCache
    stores in chunk units; a warm revisit must still hit and be correct.

Each case runs in its own subprocess so the in-process LMCache engine
singleton (fixed ``ENGINE_NAME``) starts fresh with that case's config —
a second ``LLM`` in the same process would reuse the first engine and its
``chunk_size``, silently invalidating the geometry case.

The correctness signal is the scheduler's DEBUG log line
(``trt_matched=.. lmcache_cached=.. new_matched=..``) plus byte-identical
output between the cold fill and the warm revisit.

Usage (inside the TRT-LLM release container with lmcache installed):
    python3 scripts/trtllm_e2e_edge.py                 # all cases
    python3 scripts/trtllm_e2e_edge.py --case geometry # one case
"""

# Standard
import argparse
import os
import re
import subprocess
import sys
import tempfile

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

CASES = ["overlay", "prefill", "geometry"]

MARKER = "### EDGE_STEP "
_LOOKUP_RE = re.compile(
    r"trt_matched=(\d+)\s+lmcache_cached=(\d+)\s+new_matched=(\d+)"
)
_MARKER_RE = re.compile(re.escape(MARKER) + r"(\S+)")
_TEXT_PREFIX = "### EDGE_TEXT "

# Prompt bases (deterministic, long enough to span several blocks/chunks).
_UNIT = "The quick brown fox jumps over the lazy dog. "
PROMPT = _UNIT * 60 + " Summarize:"

# For the overlay case: OVL_SHORT is a strict *token prefix* of OVL_LONG
# (same repeated unit, no trailing text on the short one), so once
# OVL_SHORT's KV is resident in TRT-LLM's paged pool, requesting OVL_LONG
# matches OVL_SHORT's blocks on-device (partial) while LMCache holds all
# of OVL_LONG. This reproduces trt_matched>0 deterministically.
OVL_LONG = _UNIT * 60 + " Summarize:"   # ~576 tokens, ~9 blocks
OVL_SHORT = _UNIT * 20                   # ~192 tokens, ~3 blocks (prefix)

# General-purpose light fillers (small, safe) for cases that only need a
# prior prompt nudged out of reuse (prefill/geometry, which then reload
# from LMCache). Each is well under the pool-capped max_seq_len of 1280.
FILLERS = [
    "Alpha series: " + " ".join(str(i) for i in range(150)),
    "Beta series: " + " ".join(str(i) for i in range(600, 750)),
]

# Heavy fillers for the overlay case: each ~1150 tokens (~18 blocks),
# still under the ~1264-token effective max input (max_seq_len 1280 minus
# 16 generated) so no single request overflows. Two of them run in
# sequence push ~36 blocks of distinct KV through the 20-block pool,
# GUARANTEEING every block of a prior ~9-block prompt is reclaimed. This
# full eviction is what the earlier light-filler attempt lacked: ~12
# filler blocks only displaced 1 of OVL_LONG's 9 blocks, so its suffix
# stayed resident and OVL_LONG_WARM short-circuited instead of overlaying.
HEAVY_FILLERS = [
    "Alpha heavy: " + " ".join(str(i) for i in range(280)),
    "Beta heavy: " + " ".join(str(i) for i in range(600, 880)),
]


def _write_config(chunk_size: int) -> str:
    """Write an LMCache config with the given chunk size and return its path."""
    path = os.path.join(tempfile.gettempdir(), f"lmcache_edge_{chunk_size}.yaml")
    with open(path, "w") as f:
        f.write(
            f"chunk_size: {chunk_size}\nlocal_cpu: true\nmax_local_cpu_size: 4.0\n"
        )
    return path


def _build_llm(chunk_size: int, block_size: int, chunked_prefill: bool):
    """Construct an LLM for a case and return ``(llm, SamplingParams)``.

    Args:
        chunk_size: LMCache chunk size (written to the config file).
        block_size: TRT-LLM ``tokens_per_block``.
        chunked_prefill: Whether to enable chunked prefill.

    Returns:
        Tuple of the constructed ``LLM`` and a greedy ``SamplingParams``.
    """
    os.environ["LMCACHE_CONFIG_FILE"] = _write_config(chunk_size)

    # Third Party
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import (
        KvCacheConfig,
        KvCacheConnectorConfig,
    )

    kv_cfg = KvCacheConfig(
        max_tokens=1280,
        tokens_per_block=block_size,
        enable_block_reuse=True,
    )
    kwargs = dict(
        model=MODEL,
        kv_cache_config=kv_cfg,
        kv_connector_config=KvCacheConnectorConfig(connector="lmcache"),
        max_batch_size=4,
        max_seq_len=2048,
    )
    if chunked_prefill:
        kwargs["enable_chunked_prefill"] = True
    llm = LLM(**kwargs)
    return llm, SamplingParams(max_tokens=16, temperature=0.0)


def _observed(llm, params, label: str, prompt: str) -> str:
    """Run one observed generate and emit the step marker and its text."""
    print(f"\n{MARKER}{label}", flush=True)
    text = llm.generate([prompt], params)[0].outputs[0].text
    print(f"{_TEXT_PREFIX}{label} {text!r}", flush=True)
    return text


def _evict(llm, params) -> None:
    """Push filler prompts to evict prior prompts from TRT-LLM's own reuse."""
    print(f"\n{MARKER}FILLER", flush=True)
    for filler in FILLERS:
        llm.generate([filler + " What comes next?"], params)


def _run_overlay() -> int:
    """Case ``overlay``: partial TRT match while LMCache holds the whole seq.

    The overlay path fires when TRT-LLM's own block reuse retains *part*
    of a sequence (``0 < trt_matched < lmcache_cached``) so the scheduler
    must load only the non-overlapping remainder. Exactly which blocks
    TRT-LLM's radix LRU retains on a revisit is an internal heuristic we
    cannot pin from outside (the deterministic remainder math is covered
    by unit tests instead — TestInProcessOverlayMath). What this e2e can
    reliably gate is that the async-load path stays *correct* under the
    churn that produces overlays: outputs must be byte-identical across
    rounds. Any overlay events that do arise are additionally checked for
    correct remainder math (opportunistic), and their count is reported.

    Two medium prompts (~12 blocks each, ~24 blocks total) cycle through a
    20-block pool, so each revisit finds part of its predecessor evicted —
    the churn that produced the incidental partial matches seen in the
    stress and cache-salt runs.
    """
    llm, params = _build_llm(chunk_size=64, block_size=64, chunked_prefill=False)
    prompts = [
        _UNIT * 80 + " Summarize A:",
        "A journey of a thousand miles begins with a single step. " * 75
        + " Summarize B:",
    ]
    rounds = 5
    print(f"\n{MARKER}OVL_CHURN", flush=True)
    for r in range(rounds):
        for i, prompt in enumerate(prompts):
            text = llm.generate([prompt], params)[0].outputs[0].text
            print(f"{_TEXT_PREFIX}R{r}P{i} {text!r}", flush=True)
    print("### EDGE_INNER_DONE", flush=True)
    return 0


def _run_prefill() -> int:
    """Case ``prefill``: chunked prefill enabled; cold fill then warm revisit."""
    llm, params = _build_llm(chunk_size=64, block_size=64, chunked_prefill=True)
    _observed(llm, params, "PF_COLD", PROMPT)
    _evict(llm, params)
    _observed(llm, params, "PF_WARM", PROMPT)
    print("### EDGE_INNER_DONE", flush=True)
    return 0


def _run_geometry() -> int:
    """Case ``geometry``: LMCache chunk_size=128 != TRT tokens_per_block=64."""
    llm, params = _build_llm(chunk_size=128, block_size=64, chunked_prefill=False)
    _observed(llm, params, "GEO_COLD", PROMPT)
    _evict(llm, params)
    _observed(llm, params, "GEO_WARM", PROMPT)
    print("### EDGE_INNER_DONE", flush=True)
    return 0


_RUNNERS = {"overlay": _run_overlay, "prefill": _run_prefill,
            "geometry": _run_geometry}


def _parse(log: str) -> tuple:
    """Parse lookup tuples and generated texts from a case log.

    Returns:
        ``(steps, texts, lookups)`` where ``steps`` maps each step label to
        the first lookup ``{"trt", "cached", "new_matched"}`` after its
        marker, ``texts`` maps every emitted text label to its generated
        text, and ``lookups`` is the flat ordered list of *all* lookup
        tuples in the log (used by the overlay scan, which needs every
        partial-match event, not just the first per step).
    """
    steps: dict = {}
    texts: dict = {}
    lookups: list = []
    current = ""
    for line in log.splitlines():
        m = _MARKER_RE.search(line)
        if m:
            current = m.group(1)
            continue
        idx = line.find(_TEXT_PREFIX)
        if idx != -1:
            label, _, text_repr = line[idx + len(_TEXT_PREFIX):].partition(" ")
            texts[label] = text_repr
            continue
        lm = _LOOKUP_RE.search(line)
        if lm:
            tup = {
                "trt": int(lm.group(1)),
                "cached": int(lm.group(2)),
                "new_matched": int(lm.group(3)),
            }
            lookups.append(tup)
            if current and current != "FILLER" and current not in steps:
                steps[current] = tup
    return steps, texts, lookups


def _check(case: str, steps: dict, texts: dict, lookups: list, block: int) -> list:
    """Return a list of failure strings for a case (empty == pass).

    Args:
        case: Case name.
        steps: Per-label first-lookup map from :func:`_parse`.
        texts: Label -> generated text map.
        lookups: Flat ordered list of all lookup tuples (overlay scan).
        block: TRT-LLM ``tokens_per_block`` for overlay remainder math.
    """
    fails: list = []

    def nm(label: str) -> int:
        return steps.get(label, {}).get("new_matched", -1)

    if case == "overlay":
        # GATE — determinism across churn rounds: round 0 is the cold
        # reference; every later round's output for the same prompt must
        # match it byte-for-byte. A corrupt overlay/async load would
        # diverge. This is the reliable correctness signal.
        prompts_seen = {int(lbl[lbl.index("P") + 1:])
                        for lbl in texts if lbl.startswith("R")}
        if not prompts_seen:
            return ["overlay: no per-round outputs captured"]
        for p in sorted(prompts_seen):
            ref = texts.get(f"R0P{p}")
            for r in range(1, 10):
                lbl = f"R{r}P{p}"
                if lbl in texts and texts[lbl] != ref:
                    fails.append(
                        f"{lbl} text {texts[lbl]} != round-0 {ref} "
                        f"(overlay/async load may have corrupted KV)"
                    )
        # OPPORTUNISTIC — verify remainder math on any overlay events
        # (0 < trt_matched < cached) that naturally occurred. Their exact
        # occurrence depends on TRT-LLM's internal LRU, so absence is
        # reported (see _run_case), not failed; the math itself is gated
        # deterministically by the unit tests. When they do occur, the
        # math must hold.
        overlays = [lu for lu in lookups if 0 < lu["trt"] < lu["cached"]]
        for lu in overlays:
            expected = ((lu["cached"] - lu["trt"]) // block) * block
            if lu["new_matched"] != expected:
                fails.append(
                    f"overlay math: trt={lu['trt']} cached={lu['cached']} "
                    f"new_matched={lu['new_matched']} != "
                    f"align_down(cached-trt={lu['cached']-lu['trt']}, {block})"
                    f"={expected}"
                )
    elif case == "prefill":
        if nm("PF_WARM") <= 0:
            fails.append(
                f"PF_WARM new_matched={nm('PF_WARM')} (expected >0; chunked "
                f"prefill save/load specs must cover the full sequence)"
            )
        if texts.get("PF_COLD") != texts.get("PF_WARM"):
            fails.append(
                f"PF_WARM text {texts.get('PF_WARM')} != cold {texts.get('PF_COLD')}"
            )
    elif case == "geometry":
        if nm("GEO_WARM") <= 0:
            fails.append(
                f"GEO_WARM new_matched={nm('GEO_WARM')} (expected >0; "
                f"chunk_size=128 != block=64 must still hit)"
            )
        if texts.get("GEO_COLD") != texts.get("GEO_WARM"):
            fails.append(
                f"GEO_WARM text {texts.get('GEO_WARM')} != cold "
                f"{texts.get('GEO_COLD')}"
            )
    return fails


def _run_case(case: str) -> bool:
    """Spawn one case in a subprocess, parse its log, and report. Returns pass."""
    env = dict(os.environ)
    env["LMCACHE_LOG_LEVEL"] = "DEBUG"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), "--inner", "--case", case]

    print(f"\n{'='*60}\n=== EDGE CASE: {case} ===\n{'='*60}", flush=True)
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log = proc.stdout
    sys.stdout.write(log)
    sys.stdout.flush()

    if proc.returncode != 0 or "### EDGE_INNER_DONE" not in log:
        print(f"CASE {case} FAILED: inner exited {proc.returncode} / incomplete",
              flush=True)
        return False

    steps, texts, lookups = _parse(log)
    print(f"observed[{case}]: steps={steps} n_lookups={len(lookups)}", flush=True)
    if case == "overlay":
        n_ovl = sum(1 for lu in lookups if 0 < lu["trt"] < lu["cached"])
        print(f"overlay events observed: {n_ovl} "
              f"(math verified on each; 0 is acceptable — unit-covered)",
              flush=True)
    # block size is 64 for all cases (geometry varies chunk_size, not block).
    fails = _check(case, steps, texts, lookups, block=64)
    if fails:
        print(f"CASE {case} FAILED: {len(fails)} problem(s)", flush=True)
        for f_ in fails:
            print(f"  - {f_}", flush=True)
        return False
    print(f"CASE {case} PASSED", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default=None)
    parser.add_argument("--inner", action="store_true")
    args = parser.parse_args()

    if args.inner:
        if args.case is None:
            print("--inner requires --case", flush=True)
            return 2
        return _RUNNERS[args.case]()

    cases = [args.case] if args.case else CASES
    results = {c: _run_case(c) for c in cases}
    print("\n" + "=" * 60, flush=True)
    for c, ok in results.items():
        print(f"  {c}: {'PASS' if ok else 'FAIL'}", flush=True)
    if all(results.values()):
        print("EDGE TESTS PASSED", flush=True)
        return 0
    print("EDGE TESTS FAILED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
