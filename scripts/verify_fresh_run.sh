#!/usr/bin/env bash
# verify_fresh_run.sh — prove a clean checkout reaches a running server using
# ONLY uv (no Docker, no devcontainer). This is the spec §16 Docker replacement
# and the T8 [GATE].
#
# What it does:
#   1. Export the current HEAD to a scratch dir (a pristine "fresh clone").
#   2. `uv sync --extra <extra> --group dev` in that dir (isolated .venv).
#   3. Boot `uv run kuaa serve` on a free port.
#   4. Poll GET /health until 200 (or fail after a timeout).
#   5. Tear the server down; clean up; report PASS/FAIL.
#
# Usage:  scripts/verify_fresh_run.sh [--keep] [--port N] [--timeout SECONDS] [--extra EXTRA]
#   --keep      leave the scratch checkout + venv for inspection
#   --port      health-check port (default: 8599, avoids a dev server on 8501)
#   --timeout   seconds to wait for /health (default: 180; cold uv sync is slow)
#   --extra     uv extras to install (default: web)
#               "web" avoids the torch/model wheels and the faster-whisper lock
#               inconsistency — /health is a liveness probe and only needs the
#               FastAPI import chain (fastapi, uvicorn, jinja2, etc.), NOT torch.
#               All model imports in this codebase are lazy (inside functions).
#               If you need to verify the full ML pipeline syncs cleanly, pass
#               "--extra full", but be aware that:
#                 (a) it downloads multi-GB wheels (torch, transformers, etc.), and
#                 (b) the committed uv.lock has a pre-existing faster-whisper/av
#                     resolution inconsistency that the maintainer must fix before
#                     `uv sync --extra full` can complete in a fresh env.
#               The default "web" extra boots /health reliably and is the intended
#               CI usage until the faster-whisper lock is reconciled.
#
# Note: git archive exports exactly what `git clone` + checkout would give —
# no .git, no .venv, no config/local.yaml (gitignored). The app must boot on
# config/default.yaml alone. This is the correct "fresh clone" precondition.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PORT=8599
TIMEOUT=180
KEEP=0
EXTRA="web"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep) KEEP=1; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --extra) EXTRA="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null 2>&1 || {
  echo "FAIL: uv not installed (https://docs.astral.sh/uv/)" >&2
  exit 1
}

WORK="$(mktemp -d -t kuaa-fresh-XXXXXX)"
SERVER_PID=""

cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" >/dev/null 2>&1 || true
  [[ -n "$SERVER_PID" ]] && wait "$SERVER_PID" 2>/dev/null || true
  if [[ "$KEEP" -eq 0 ]]; then
    rm -rf "$WORK"
  else
    echo "kept: $WORK"
  fi
}
trap cleanup EXIT

echo "==> 1/4  exporting HEAD to a pristine checkout: $WORK"
# git archive gives a clean tree — no .git, no .venv, no config/local.yaml
git -C "$REPO_ROOT" archive --format=tar HEAD | tar -x -C "$WORK"

echo "==> 2/4  uv sync --extra ${EXTRA} --group dev (cold; first run may take a few minutes)"
( cd "$WORK" && uv sync --extra "$EXTRA" --group dev )

echo "==> 3/4  booting: uv run kuaa serve --host 127.0.0.1 --port ${PORT} --no-reload"
( cd "$WORK" && uv run kuaa serve --host 127.0.0.1 --port "$PORT" --no-reload ) \
  > "$WORK/server.log" 2>&1 &
SERVER_PID=$!

echo "==> 4/4  polling http://127.0.0.1:${PORT}/health (timeout ${TIMEOUT}s)"
deadline=$(( $(date +%s) + TIMEOUT ))
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server process exited before /health was ready. Log tail:" >&2
    tail -40 "$WORK/server.log" >&2 || true
    exit 1
  fi
  if [[ "$(date +%s)" -ge "$deadline" ]]; then
    echo "FAIL: /health did not return 200 within ${TIMEOUT}s. Log tail:" >&2
    tail -40 "$WORK/server.log" >&2 || true
    exit 1
  fi
  sleep 2
done

body="$(curl -fsS "http://127.0.0.1:${PORT}/health")"
echo "PASS: fresh checkout booted via uv and /health responded: ${body}"
