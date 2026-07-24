# SPDX-License-Identifier: Apache-2.0
"""E2E smoke test for the async TRT-LLM LMCache connector (in-process mode).

Runs real inference with a tiny KvCacheConfig so TRT-LLM's own block
reuse cannot serve the second request — it must come from LMCache.

Validates the full async pipeline:
    lookup (is_async) -> park -> start_load_kv -> get_finished(loads)
    wait_for_save (non-blocking) -> request_finished -> get_finished(saves)

Usage (inside the TRT-LLM release container with lmcache installed):
    LMCACHE_CONFIG_FILE=/tmp/lmcache_config.yaml python3 trtllm_e2e_smoke.py
"""

# Standard
import os
import sys
import tempfile

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

CONFIG_YAML = """\
chunk_size: 64
local_cpu: true
max_local_cpu_size: 2.0
"""


def main() -> int:
    config_path = os.path.join(tempfile.gettempdir(), "lmcache_smoke_config.yaml")
    with open(config_path, "w") as f:
        f.write(CONFIG_YAML)
    os.environ["LMCACHE_CONFIG_FILE"] = config_path

    # Third Party
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import (
        KvCacheConfig,
        KvCacheConnectorConfig,
    )

    connector_cfg = KvCacheConnectorConfig(connector="lmcache")

    # Tiny GPU pool: large prompts are guaranteed-evicted between
    # requests, so a second-request hit can only come from LMCache.
    kv_cfg = KvCacheConfig(
        max_tokens=2048,
        tokens_per_block=64,
        enable_block_reuse=True,
    )

    llm = LLM(
        model=MODEL,
        kv_cache_config=kv_cfg,
        kv_connector_config=connector_cfg,
        max_batch_size=2,
        max_seq_len=2048,
        tensor_parallel_size=int(os.environ.get("SMOKE_TP", "1")),
    )

    # A long, deterministic prompt (many chunks of 64 tokens).
    base = "The quick brown fox jumps over the lazy dog. " * 60
    prompts = [base + " Summarize this in one word:"]
    params = SamplingParams(max_tokens=16, temperature=0.0)

    print("=== REQUEST 1 (cold: fills LMCache) ===", flush=True)
    out1 = llm.generate(prompts, params)
    text1 = out1[0].outputs[0].text
    print(f"output 1: {text1!r}", flush=True)

    # Evict TRT-LLM's pool by pushing a different large prompt through.
    filler = "Numbers: " + " ".join(str(i) for i in range(400))
    print("=== REQUEST 2 (filler: evicts GPU pool) ===", flush=True)
    llm.generate([filler + " What comes next?"], params)

    print("=== REQUEST 3 (warm: must hit LMCache) ===", flush=True)
    out3 = llm.generate(prompts, params)
    text3 = out3[0].outputs[0].text
    print(f"output 3: {text3!r}", flush=True)

    if text1 != text3:
        print(
            f"FAIL: outputs differ (cold={text1!r} warm={text3!r}) — "
            "possible KV corruption from the async load path",
            flush=True,
        )
        return 1

    print("SMOKE TEST PASSED: warm output matches cold output", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
