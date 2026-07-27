#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Engine-restart persistence e2e for the MP TRT-LLM adapter.
#
# Starts ONE long-lived LMCache MP cache server, then drives two
# SEPARATE TRT-LLM engine processes against it: the first fills the cache
# and exits (its GPU pool and block reuse die with it), the second starts
# fresh and must serve prompt P from the persistent server. Proves the
# cache survives a full engine teardown. Run inside the TRT-LLM release
# container with lmcache installed and at least 1 GPU visible.
set -uo pipefail

SERVER_LOG=/workspace/LMCache/mp_persist_server.log
SERVER_URL="tcp://localhost:5555"

echo "### starting LMCache MP server ###"
python3 -m lmcache.v1.multiprocess.server \
    --host localhost --port 5555 \
    --chunk-size 64 \
    --l1-size-gb 4 \
    --eviction-policy LRU \
    --max-workers 2 \
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

python3 scripts/trtllm_mp_persist.py --server-url "$SERVER_URL"
RC=$?

echo "### stopping MP server ###"
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null

echo "MP_PERSIST_E2E_EXIT=$RC"
exit $RC
