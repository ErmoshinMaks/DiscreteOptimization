#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


TESTS = [
    "fl_25_2",
    "fl_100_1",
    "fl_200_7",
    "fl_500_7",
    "fl_1000_2",
    "fl_2000_2",
]

TIMEOUT_SEC = 620


def load_instance(path: Path):
    lines = path.read_text().strip().splitlines()
    n, m = map(int, lines[0].split())

    s = [0.0] * n
    cap = [0.0] * n
    fx = [0.0] * n
    fy = [0.0] * n
    for i in range(n):
        a = list(map(float, lines[1 + i].split()))
        s[i], cap[i], fx[i], fy[i] = a[0], a[1], a[2], a[3]

    dem = [0.0] * m
    cx = [0.0] * m
    cy = [0.0] * m
    for i in range(m):
        a = list(map(float, lines[1 + n + i].split()))
        dem[i], cx[i], cy[i] = a[0], a[1], a[2]

    return n, m, s, cap, fx, fy, dem, cx, cy


def true_objective(n, m, s, cap, fx, fy, dem, cx, cy, assign1: List[int]) -> float:
    assign = [x - 1 for x in assign1]
    used = set(assign)
    fixed = sum(s[j] for j in used)
    transport = 0.0
    load = [0.0] * n

    for i in range(m):
        j = assign[i]
        transport += math.hypot(cx[i] - fx[j], cy[i] - fy[j])
        load[j] += dem[i]

    for j in range(n):
        if load[j] > cap[j] + 1e-6:
            raise ValueError(f"перегрузка магазина {j + 1}: {load[j]} > {cap[j]}")

    return fixed + transport


def verify(
    n: int,
    m: int,
    s,
    cap,
    fx,
    fy,
    dem,
    cx,
    cy,
    claimed: float,
    assign: List[int],
) -> Tuple[bool, str]:
    if len(assign) != m:
        return False, f"ожидалось {m} назначений, получено {len(assign)}"

    for i, x in enumerate(assign):
        if x < 1 or x > n:
            return False, f"клиент {i + 1}: недопустимый магазин {x}"

    try:
        val = true_objective(n, m, s, cap, fx, fy, dem, cx, cy, assign)
    except ValueError as err:
        return False, str(err)

    tol = max(1e-3, 1e-7 * max(abs(val), abs(claimed)))
    if abs(val - claimed) > tol:
        return False, f"цель не сходится: заявлено {claimed}, по данным {val}"
    return True, "OK"


def run_solver(task_dir: Path, test_path: Path) -> Tuple[Optional[float], Optional[List[int]], str]:
    run_sh = task_dir / "run.sh"
    try:
        proc = subprocess.run(
            [str(run_sh), str(test_path)],
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return None, None, f"таймаут {TIMEOUT_SEC} сек"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0 and err:
        return None, None, f"код {proc.returncode}: {err[:500]}"

    lines = out.splitlines()
    if not lines:
        return None, None, err or "пустой вывод"

    try:
        claimed = float(lines[0].strip())
    except ValueError:
        return None, None, f"первая строка должна быть числом: {lines[0][:80]}"

    if len(lines) < 2:
        return claimed, None, "нет второй строки с назначениями"

    try:
        assign = [int(x) for x in lines[1].split()]
    except ValueError:
        return None, None, "вторая строка должна содержать целые номера магазинов"

    return claimed, assign, err


def main() -> int:
    task_dir = Path(__file__).resolve().parent
    data_dir = task_dir / "data"
    run_sh = task_dir / "run.sh"

    if not run_sh.exists():
        print("Ошибка: нет файла run.sh", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print("Ошибка: нет каталога data/", file=sys.stderr)
        return 1

    all_ok = True
    for name in TESTS:
        test_path = data_dir / name
        if not test_path.exists():
            print(f"[{name}] нет файла теста", file=sys.stderr)
            all_ok = False
            print(f"{name}\tFAIL")
            continue

        n, m, s, cap, fx, fy, dem, cx, cy = load_instance(test_path)
        claimed, assign, msg = run_solver(task_dir, test_path)
        if assign is None or len(assign) != m:
            print(f"[{name}] ошибка запуска: {msg}", file=sys.stderr)
            all_ok = False
            print(f"{name}\tFAIL")
            continue

        ok, why = verify(n, m, s, cap, fx, fy, dem, cx, cy, claimed, assign)
        if not ok:
            print(f"[{name}] {why}", file=sys.stderr)
            all_ok = False
            print(f"{name}\tFAIL")
            continue

        print(f"{name}\t{claimed:.6f}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
