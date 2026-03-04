#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

TESTS = [
    "ks_30_0",
    "ks_50_0",
    "ks_200_0",
    "ks_400_0",
    "ks_1000_0",
    "ks_10000_0",
]


def load_test(data_dir: Path, name: str):
    path = data_dir / name
    with open(path) as f:
        lines = f.read().strip().split("\n")
    n, W = map(int, lines[0].split())
    items = []
    for i in range(1, 1 + n):
        v, w = map(int, lines[i].split())
        items.append((v, w))
    return n, W, items


def verify(n: int, W: int, items: list, claimed_value: int, solution: list) -> Tuple[bool, str]:
    if len(solution) != n:
        return False, f"длина вектора решения {len(solution)}, ожидается {n}"
    total_v = 0
    total_w = 0
    for i, (v, w) in enumerate(items):
        if solution[i] not in (0, 1):
            return False, f"x[{i}] = {solution[i]}, ожидается 0 или 1"
        if solution[i] == 1:
            total_v += v
            total_w += w
    if total_w > W:
        return False, f"вес {total_w} > W = {W}"
    if total_v != claimed_value:
        return False, f"заявленная стоимость {claimed_value} != фактическая {total_v}"
    return True, "OK"


def run_solution(script_dir: Path, test_path: Path) -> Tuple[int, Optional[List[int]], str]:
    run_sh = script_dir / "run.sh"
    cmd = [str(run_sh), str(test_path)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=script_dir,
        )
    except subprocess.TimeoutExpired:
        return 0, None, "Timeout 300s"
    out = result.stdout
    err = result.stderr or ""
    lines = out.strip().split("\n")
    if not lines:
        return 0, None, err or "Нет вывода"
    try:
        value = int(lines[0])
    except ValueError:
        return 0, None, (err or f"Не число в первой строке: {lines[0][:50]}")
    solution = None
    if len(lines) >= 2:
        parts = lines[1].split()
        if len(parts) == 0:
            solution = []
        else:
            try:
                solution = [int(x) for x in parts]
            except ValueError:
                solution = None
    return value, solution, err


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"

    if not (script_dir / "run.sh").exists():
        print("Ошибка: run.sh не найден в каталоге чекера.", file=sys.stderr)
        sys.exit(1)
    if not data_dir.is_dir():
        print("Ошибка: каталог data/ не найден.", file=sys.stderr)
        sys.exit(1)

    all_ok = True
    for name in TESTS:
        test_path = data_dir / name
        if not test_path.exists():
            print(f"[{name}] файл не найден", file=sys.stderr)
            all_ok = False
            continue
        n, W, items = load_test(data_dir, name)
        value, solution, run_err = run_solution(script_dir, test_path)
        if run_err and "OK" not in run_err:
            print(f"[{name}] запуск: {run_err}", file=sys.stderr)
        if solution is None or len(solution) != n:
            ok = False
            msg = run_err or "нет вектора решения или неверная длина"
        else:
            ok, msg = verify(n, W, items, value, solution)
        if not ok:
            print(f"[{name}] проверка: {msg}", file=sys.stderr)
            all_ok = False
        print(f"{name}\t{value}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
