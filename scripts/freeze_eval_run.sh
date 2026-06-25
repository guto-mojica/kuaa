#!/usr/bin/env bash
# Snapshot a completed eval run's grades into a SHA256-checksummed tarball.
#
# The tarball stays outside git (per the data/eval/<run_id> gitignore policy
# in docs/EVAL_PROTOCOL.md §7). Record the printed SHA in
# docs/eval_session_log.md so future runs of the M4 ablation can prove which
# grades fed which results.
#
# Grade store layout (kuaa/eval/grades.py::EvalRun.jsonl_path):
#     data/eval/<run_id>.jsonl     (append-only JSONL written by /api/eval/grade)
# Optional per-run directory (candidate slates, summaries):
#     data/eval/<run_id>/          (gitignored runtime artefacts)
#
# Usage: scripts/freeze_eval_run.sh <run-id>
# Example: scripts/freeze_eval_run.sh m3-run-1
set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run-id> (e.g. m3-run-1)" >&2
  exit 2
fi

EVAL_ROOT="data/eval"
GRADES_JSONL="${EVAL_ROOT}/${RUN_ID}.jsonl"
RUN_DIR="${EVAL_ROOT}/${RUN_ID}"

if [[ ! -f "${GRADES_JSONL}" ]]; then
  echo "No grades JSONL at ${GRADES_JSONL} — has annotation started?" >&2
  echo "Hint: open /eval?token=<EVAL_ADMIN_TOKEN> and grade some rows first." >&2
  exit 1
fi

GRADE_LINES=$(wc -l < "${GRADES_JSONL}" | tr -d ' ')
if (( GRADE_LINES == 0 )); then
  echo "Grades JSONL ${GRADES_JSONL} is empty — nothing to freeze." >&2
  exit 1
fi

DATE_STAMP=$(date +%Y%m%d)
TARBALL="${EVAL_ROOT}/${RUN_ID}.frozen-${DATE_STAMP}.tar.gz"

# Tar the JSONL plus the run dir (if present). The run dir holds candidate
# slates + retrieval summaries; freezing it together with the grades keeps
# the snapshot self-contained.
TAR_INPUTS=("${GRADES_JSONL}")
if [[ -d "${RUN_DIR}" ]]; then
  TAR_INPUTS+=("${RUN_DIR}")
fi

tar -czf "${TARBALL}" "${TAR_INPUTS[@]}"
SHA=$(sha256sum "${TARBALL}" | awk '{print $1}')
SIZE=$(wc -c < "${TARBALL}" | tr -d ' ')

echo "Frozen: ${TARBALL}"
echo "  Grade lines:  ${GRADE_LINES}"
echo "  Tarball size: ${SIZE} bytes"
echo "  SHA256:       ${SHA}"
echo
echo "Next steps:"
echo "  1. Record the SHA in docs/eval_session_log.md (one line per run)."
echo "  2. The tarball is NOT committed (matches data/eval/*.frozen-*.tar.gz"
echo "     in .gitignore). Store it wherever your team keeps frozen eval"
echo "     snapshots (S3, archive volume, etc.)."
