#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# TP=2 e2e for the MP (multi-process) TRT-LLM adapter.
#
# Starts the LMCache MP cache server, then runs a tensor-parallel (TP=2)
# TRT-LLM instance against it and verifies a warm LMCache hit with
# byte-identical output. Run inside the TRT-LLM release container with
# lmcache installed and at least 2 GPUs visible.
#
# Exercises per-rank worker shards (worker_id=rank, sharded pool
# registration) and the multi-rank allgather in get_finished that gates
# async-load completion on ALL ranks reporting.
#
# max-workers must be >= tp_size (2) so both ranks' shards register.
# Override with MP_MAX_WORKERS=1 to serialize STORE/RETRIEVE through a
# single server thread (isolates concurrency races from device bugs).
set -uo pipefail

SERVER_LOG=/workspace/LMCache/mp_tp2_server.log
SERVER_URL="tcp://localhost:5555"
MAX_WORKERS="${MP_MAX_WORKERS:-4}"

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

python3 scripts/trtllm_mp_tp2.py --server-url "$SERVER_URL"
RC=$?

echo "### stopping MP server ###"
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null

echo "MP_TP2_E2E_EXIT=$RC"
exit $RC
