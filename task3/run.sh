#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LIMIT="${2:-580}"
exec python3 "$ROOT/solver.py" "${1:?usage: $0 <input_file> [time_limit]}" "$LIMIT"
