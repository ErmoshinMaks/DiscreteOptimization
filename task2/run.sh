#!/usr/bin/env bash
# Запуск решателя задачи о рюкзаке.
# Использование: ./run.sh [файл_теста]
# Если файл не указан — чтение из stdin.

set -e
cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    python3 knapsack.py
else
    python3 knapsack.py < "$1"
fi
