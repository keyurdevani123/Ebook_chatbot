#!/usr/bin/env bash
# ============================================================
# reindex.sh — Nightly e-book re-indexing cron script
# ============================================================
# Schedule with cron (runs at 2 AM every night):
#   0 2 * * * /path/to/ebook-backend/reindex.sh
#
# Usage:
#   chmod +x reindex.sh
#   ./reindex.sh                     # Manual run
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/reindex_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

echo "=========================================="  | tee -a "${LOG_FILE}"
echo "Nightly Re-index started: $(date --iso-8601=seconds)"  | tee -a "${LOG_FILE}"
echo "=========================================="  | tee -a "${LOG_FILE}"

python "${SCRIPT_DIR}/subprocess_runner.py" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================="  | tee -a "${LOG_FILE}"
echo "Re-index finished: $(date --iso-8601=seconds) | Exit code: ${EXIT_CODE}"  | tee -a "${LOG_FILE}"
echo "=========================================="  | tee -a "${LOG_FILE}"

exit "${EXIT_CODE}"
