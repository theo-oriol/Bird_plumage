#!/usr/bin/env bash

set -uo pipefail

# ── Settings ─────────────────────────────────────────────────────────────────
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root, whatever the caller's cwd

set -a; [ -f .env ] && . ./.env; set +a
PYTHON="${PYTHON:-python}"
A7_WORKERS=25
POLL_SECONDS=120

log() {
    echo "[$(date -Is)] $*"
}

# ── Wait for A6 ──────────────────────────────────────────────────────────────
log "wait_a6_then_build_a7: watching for build_a6_classification_cache.py to finish"
log "PID of this watcher: $$"

while pgrep -f "build_a6_classification_cache\.py" > /dev/null; do
    n=$(pgrep -f "build_a6_classification_cache\.py" | wc -l)
    log "still running ($n matching processes) - checking again in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
done

# ── Build A7 ─────────────────────────────────────────────────────────────────
log "no build_a6_classification_cache.py process found - A6 classification build is done"
log "starting: $PYTHON scripts/build_a7_cache.py --workers $A7_WORKERS"

"$PYTHON" scripts/build_a7_cache.py --workers "$A7_WORKERS"
status=$?

log "build_a7_cache.py exited with status $status"
