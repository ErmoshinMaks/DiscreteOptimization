#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ge 1 ]]; then
  python3 "${SCRIPT_DIR}/solve.py" "$1"
else
  python3 "${SCRIPT_DIR}/solve.py"
fi

