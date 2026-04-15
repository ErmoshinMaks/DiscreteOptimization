#!/usr/bin/env bash
#пример ./run.sh "data 2/fl_25_2"
set -e
cd "$(dirname "$0")"
if [ "$#" -eq 0 ]; then
  exec python3 solver.py
else
  exec python3 solver.py < "$1"
fi
