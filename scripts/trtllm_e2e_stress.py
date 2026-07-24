# SPDX-License-Identifier: Apache-2.0
"""Stress test for the async TRT-LLM LMCache connector.

Goes beyond the smoke test with:

1. Multi-round churn: N distinct long prompts cycled repeatedly through
   a GPU pool sized to hold ~1 of them. Every revisit must load from
   LMCache; outputs must stay byte-identical across all rounds.
2. Concurrent batch: a batch mixing cold prompts (async saves) and warm
   prompts (async loads) submitted together, exercising simultaneous
   in-flight saves and loads in get_finished.
3. Partial-prefix reuse: a prompt sharing a long prefix with a cached
   one, checking prefix loads compose with fresh suffix computation.

Usage (inside the TRT-LLM release container with lmcache installed):
    python3 scripts/trtllm_e2e_stress.py [--preset lmcache|lmcache-mp]
"""

# Standard
import argparse
import os
import sys
import tempfile

CONFIG_YAML = """\
chunk_size: 64
local_cpu: true
max_local_cpu_size: 4.0
"""

NUM_PROMPTS = 4
NUM_ROUNDS = 3


def _make_prompts() -> list:
    """Distinct long deterministic prompts (each ~600-700 tokens)."""
    bases = [
        "The quick brown fox jumps over the lazy dog. " * 55,
        "A journey of a thousand miles begins with a single step. " * 50,
        "To be or not to be, that is the question of the play. " * 50,
        "All that glitters is not gold, said the wise old man. " * 50,
    ]
    return [b + f" Question {i}: summarize in one word:" for i, b in enumerate(bases)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", default="lmcache", choices=["lmcache", "lmcache-mp"]
    )
    parser.add_argument("--server-url", default=None)
    args = parser.parse_args()

    config_path = os.path.join(tempfile.gettempdir(), "lmcache_stress_config.yaml")
    with open(config_path, "w") as f:
        f.write(CONFIG_YAML)
    os.environ["LMCACHE_CONFIG_FILE"] = config_path

    # Third Party
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import (
        KvCacheConfig,
        KvCacheConnectorConfig,
    )

    connector_kwargs = {"connector": args.preset}
    if args.server_url:
        connector_kwargs["server_url"] = args.server_url
    connector_cfg = KvCacheConnectorConfig(**connector_kwargs)

    # Pool holds ~1280 tokens: roughly one prompt + generation. Cycling
    # 4 prompts (~700 tokens each) guarantees eviction between rounds.
    kv_cfg = KvCacheConfig(
        max_tokens=1280,
        tokens_per_block=64,
        enable_block_reuse=True,
    )

    llm = LLM(
        model=os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        kv_cache_config=kv_cfg,
        kv_connector_config=connector_cfg,
        max_batch_size=4,
        max_seq_len=2048,
    )

    params = SamplingParams(max_tokens=12, temperature=0.0)
    prompts = _make_prompts()
    failures = []

    # --- Phase 1: multi-round churn -----------------------------------
    print("=== PHASE 1: multi-round churn ===", flush=True)
    reference: list = []
    for round_idx in range(NUM_ROUNDS):
        for i, prompt in enumerate(prompts):
            out = llm.generate([prompt], params)
            text = out[0].outputs[0].text
            if round_idx == 0:
                reference.append(text)
                print(f"round 0 prompt {i} (cold): {text!r}", flush=True)
            elif text != reference[i]:
                failures.append(
                    f"churn round {round_idx} prompt {i}: "
                    f"{text!r} != {reference[i]!r}"
                )
                print(f"round {round_idx} prompt {i}: MISMATCH", flush=True)
            else:
                print(f"round {round_idx} prompt {i}: match", flush=True)

    # --- Phase 2: concurrent batch of mixed cold/warm ------------------
    print("=== PHASE 2: concurrent mixed batch ===", flush=True)
    fresh = [
        "Space, the final frontier, these are the voyages of the ship. " * 45
        + f" New question {i}: answer briefly:"
        for i in range(2)
    ]
    # Batch = 2 warm (must async-load) + 2 cold (must async-save).
    batch = [prompts[0], prompts[1], fresh[0], fresh[1]]
    outs = llm.generate(batch, params)
    batch_texts = [o.outputs[0].text for o in outs]
    for i in range(2):
        if batch_texts[i] != reference[i]:
            failures.append(
                f"batch warm prompt {i}: {batch_texts[i]!r} != {reference[i]!r}"
            )
            print(f"batch warm prompt {i}: MISMATCH", flush=True)
        else:
            print(f"batch warm prompt {i}: match", flush=True)
    fresh_ref = batch_texts[2:]
    print(f"batch cold outputs: {fresh_ref!r}", flush=True)

    # Re-run the fresh prompts warm; they were saved during the batch.
    outs2 = llm.generate(fresh, params)
    for i, o in enumerate(outs2):
        text = o.outputs[0].text
        if text != fresh_ref[i]:
            failures.append(
                f"batch-saved prompt {i} warm: {text!r} != {fresh_ref[i]!r}"
            )
            print(f"batch-saved prompt {i} warm: MISMATCH", flush=True)
        else:
            print(f"batch-saved prompt {i} warm: match", flush=True)

    # --- Phase 3: partial-prefix reuse ---------------------------------
    print("=== PHASE 3: partial-prefix reuse ===", flush=True)
    extended = prompts[0].replace(
        "Question 0: summarize in one word:",
        "Question 0-extended: instead, summarize in exactly two words:",
    )
    out_ext = llm.generate([extended], params)
    print(f"extended output: {out_ext[0].outputs[0].text!r}", flush=True)
    # Correctness bar: it must complete without hanging or corrupting;
    # rerun it warm and require determinism.
    out_ext2 = llm.generate([extended], params)
    if out_ext2[0].outputs[0].text != out_ext[0].outputs[0].text:
        failures.append(
            f"prefix-reuse warm: {out_ext2[0].outputs[0].text!r} "
            f"!= {out_ext[0].outputs[0].text!r}"
        )
        print("prefix-reuse warm: MISMATCH", flush=True)
    else:
        print("prefix-reuse warm: match", flush=True)

    print("=" * 60, flush=True)
    if failures:
        print(f"STRESS TEST FAILED: {len(failures)} mismatches", flush=True)
        for f_ in failures:
            print(f"  - {f_}", flush=True)
        return 1
    print("STRESS TEST PASSED: all outputs deterministic across rounds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
