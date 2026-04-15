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
    for j in range(m):
        a = list(map(float, lines[1 + n + j].split()))
        dem[j], cx[j], cy[j] = a[0], a[1], a[2]
    return n, m, s, cap, fx, fy, dem, cx, cy


def true_objective(n, m, s, cap, fx, fy, dem, cx, cy, assign0: List[int]) -> float:
    assign = [j - 1 for j in assign0]
    used = set(assign)
    fix = sum(s[j] for j in used)
    trans = 0.0
    load = [0.0] * n
    for i in range(m):
        j = assign[i]
        trans += math.hypot(cx[i] - fx[j], cy[i] - fy[j])
        load[j] += dem[i]
    for j in range(n):
        if load[j] > cap[j] + 1e-6:
            raise ValueError(f"перегруз магазина {j+1}: {load[j]} > {cap[j]}")
    return fix + trans


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
    for i, v in enumerate(assign):
        if v < 1 or v > n:
            return False, f"клиент {i+1}: недопустимый магазин {v}"
    try:
        val = true_objective(n, m, s, cap, fx, fy, dem, cx, cy, assign)
    except ValueError as e:
        return False, str(e)
    if abs(val - claimed) > max(1e-3, 1e-7 * max(abs(val), abs(claimed))):
        return False, f"цель не сходится: заявлено {claimed}, по данным {val}"
    return True, "OK"


def run_solver(script_dir: Path, test_path: Path) -> Tuple[Optional[float], Optional[List[int]], str]:
    run_sh = script_dir / "run.sh"
    try:
        proc = subprocess.run(
            [str(run_sh), str(test_path)],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return None, None, f"таймаут {TIMEOUT_SEC} с"
    err = (proc.stderr or "").strip()
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 and err:
        return None, None, f"code {proc.returncode}: {err[:500]}"
    lines = out.splitlines()
    if not lines:
        return None, None, err or "пустой вывод"
    try:
        claimed = float(lines[0].strip())
    except ValueError:
        return None, None, f"первая строка не число: {lines[0][:80]}"
    assign = None
    if len(lines) >= 2:
        parts = lines[1].split()
        try:
            assign = [int(x) for x in parts]
        except ValueError:
            return None, None, "вторая строка: не целые номера магазинов"
    return claimed, assign, err


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data 2"
    if not (script_dir / "run.sh").exists():
        print("Нет run.sh", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print("Нет каталога data 2/", file=sys.stderr)
        return 1

    ok_all = True
    for name in TESTS:
        p = data_dir / name
        if not p.exists():
            print(f"[{name}] нет файла", file=sys.stderr)
            ok_all = False
            continue
        n, m, s, cap, fx, fy, dem, cx, cy = load_instance(p)
        claimed, assign, msg = run_solver(script_dir, p)
        if assign is None or len(assign) != m:
            print(f"[{name}] запуск: {msg}", file=sys.stderr)
            ok_all = False
            print(f"{name}\tFAIL")
            continue
        ok, why = verify(n, m, s, cap, fx, fy, dem, cx, cy, claimed, assign)
        if not ok:
            print(f"[{name}] {why}", file=sys.stderr)
            ok_all = False
            print(f"{name}\tFAIL")
        else:
            print(f"{name}\t{claimed}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
