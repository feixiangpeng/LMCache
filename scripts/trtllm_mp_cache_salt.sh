#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Cache-salt isolation e2e for the MP (multi-process) TRT-LLM adapter.
#
# Starts the LMCache MP cache server in the background, then runs the
# cache-salt isolation test with the lmcache-mp connector preset against
# it. Run inside the TRT-LLM release container with lmcache installed.
#
# Proves IPCCacheServerKey.cache_salt partitions the MP server's cache:
# tenant-b must not reuse tenant-a's KV for identical tokens.
set -uo pipefail

SERVER_LOG=/workspace/LMCache/mp_salt_server.log
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

python3 scripts/trtllm_e2e_cache_salt.py \
    --preset lmcache-mp --server-url "$SERVER_URL"
RC=$?

echo "### stopping MP server ###"
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null

echo "MP_SALT_E2E_EXIT=$RC"
exit $RC
