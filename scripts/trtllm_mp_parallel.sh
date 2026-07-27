#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Model-parallel e2e for the MP (multi-process) TRT-LLM adapter.
#
# Starts the LMCache MP cache server, then runs a TP*PP TRT-LLM instance
# against it and verifies a warm LMCache hit with byte-identical output.
# Run inside the TRT-LLM release container with lmcache installed and at
# least TP*PP GPUs visible.
#
# Exercises per-rank worker shards (worker_id=rank, sharded pool
# registration), the CUDA-IPC per-device import (each rank opens its pool
# on its own physical device -- TP>=4 forces cuda:1/2/3 mappings), and
# the multi-rank allgather in get_finished that gates async-load
# completion on ALL ranks reporting.
#
# Env:
#   TP               tensor-parallel degree (default 4)
#   PP               pipeline-parallel degree (default 1)
#   MP_MAX_WORKERS   server worker threads (default: max(TP*PP, 4))
#   SMOKE_MODEL      HF model id (default in the .py; needs >=TP KV heads)
#   ALLOW_TEXT_DRIFT if "1", pass --allow-text-drift (for models that are
#                    not bit-exact across the reload round-trip)
set -uo pipefail

TP="${TP:-4}"
PP="${PP:-1}"
WORLD=$((TP * PP))
DEFAULT_WORKERS=$(( WORLD > 4 ? WORLD : 4 ))
MAX_WORKERS="${MP_MAX_WORKERS:-$DEFAULT_WORKERS}"

SERVER_LOG=/workspace/LMCache/mp_parallel_server.log
SERVER_URL="tcp://localhost:5555"

echo "### starting LMCache MP server (max-workers=$MAX_WORKERS) ###"
python3 -m lmcache.v1.multiprocess.server \
    --host localhost --port 5555 \
    --chunk-size 64 \
    --l1-size-gb 4 \
    --eviction-policy LRU \
    --max-workers "$MAX_WORKERS" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for the server to bind.
for i in $(seq 1 60); do
    if grep -q "is running on" "$SERVER_LOG" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "MP SERVER DIED DURING STARTUP:"
        tail -30 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done
echo "### MP server up (pid $SERVER_PID) ###"

DRIFT_FLAG=""
if [ "${ALLOW_TEXT_DRIFT:-0}" = "1" ]; then
    DRIFT_FLAG="--allow-text-drift"
fi

echo "### running TP=$TP PP=$PP (world=$WORLD) ###"
python3 scripts/trtllm_mp_parallel.py --server-url "$SERVER_URL" \
    --tp "$TP" --pp "$PP" $DRIFT_FLAG
RC=$?

echo "### stopping MP server ###"
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null

echo "MP_PARALLEL_E2E_EXIT=$RC"
exit $RC
