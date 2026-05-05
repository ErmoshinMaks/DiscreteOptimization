#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path


TESTS = [
    "vrp_16_3_1",
    "vrp_26_8_1",
    "vrp_51_5_1",
    "vrp_101_10_1",
    "vrp_200_16_1",
    "vrp_121_7_1",
]
TIMEOUT_SEC = 620


def parse_instance(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    n, v, cap = map(int, lines[0].split())
    demand = [0] * n
    x = [0.0] * n
    y = [0.0] * n
    for i in range(n):
        d, xi, yi = lines[1 + i].split()
        demand[i] = int(d)
        x[i] = float(xi)
        y[i] = float(yi)
    return n, v, cap, demand, x, y


def dist(i: int, j: int, x: list[float], y: list[float]) -> float:
    return math.hypot(x[i] - x[j], y[i] - y[j])


def parse_output(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("пустой вывод")
    header = lines[0].split()
    if not header:
        raise ValueError("первая строка пустая")
    try:
        claimed = float(header[0])
    except ValueError as err:
        raise ValueError(f"первая строка должна начинаться с числа: {lines[0]}") from err
    routes = []
    for ln in lines[1:]:
        try:
            route = [int(v) for v in ln.split()]
        except ValueError as err:
            raise ValueError(f"маршрут должен содержать целые числа: {ln}") from err
        routes.append(route)
    return claimed, routes


def evaluate(
    n: int,
    v: int,
    cap: int,
    demand: list[int],
    x: list[float],
    y: list[float],
    claimed: float,
    routes: list[list[int]],
) -> tuple[bool, str, float]:
    if len(routes) != v:
        return False, f"ожидалось {v} маршрутов, получено {len(routes)}", 0.0

    seen = [0] * n
    total = 0.0
    for idx, route in enumerate(routes):
        if len(route) < 2:
            return False, f"маршрут {idx + 1}: слишком короткий", 0.0
        if route[0] != 0 or route[-1] != 0:
            return False, f"маршрут {idx + 1}: должен начинаться и заканчиваться 0", 0.0

        load = 0
        prev = route[0]
        for cur in route[1:]:
            if cur < 0 or cur >= n:
                return False, f"маршрут {idx + 1}: недопустимая вершина {cur}", 0.0
            total += dist(prev, cur, x, y)
            prev = cur
            if cur != 0:
                seen[cur] += 1
                load += demand[cur]
        if load > cap:
            return False, f"маршрут {idx + 1}: перегруз {load} > {cap}", 0.0

    for c in range(1, n):
        if seen[c] != 1:
            return False, f"клиент {c}: посещений {seen[c]} (должно быть 1)", 0.0

    tol = max(1e-3, 1e-7 * max(abs(total), abs(claimed)))
    if abs(total - claimed) > tol:
        return False, f"цель не сходится: заявлено {claimed}, вычислено {total}", total
    return True, "OK", total


def run_solver(task_dir: Path, test_path: Path) -> tuple[str | None, str]:
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
        return None, f"таймаут {TIMEOUT_SEC} сек"

    out = proc.stdout or ""
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return None, f"код {proc.returncode}: {err[:400]}"
    return out, err


def main() -> int:
    task_dir = Path(__file__).resolve().parent
    data_dir = task_dir / "data"
    run_sh = task_dir / "run.sh"

    if not run_sh.exists():
        print("Ошибка: не найден run.sh", file=sys.stderr)
        return 1
    if not data_dir.exists():
        print("Ошибка: не найден каталог data", file=sys.stderr)
        return 1

    all_ok = True
    for name in TESTS:
        p = data_dir / name
        if not p.exists():
            print(f"{name}\tFAIL")
            print(f"[{name}] файл теста отсутствует", file=sys.stderr)
            all_ok = False
            continue

        n, v, cap, demand, x, y = parse_instance(p)
        out, msg = run_solver(task_dir, p)
        if out is None:
            print(f"{name}\tFAIL")
            print(f"[{name}] ошибка запуска: {msg}", file=sys.stderr)
            all_ok = False
            continue

        try:
            claimed, routes = parse_output(out)
        except ValueError as err:
            print(f"{name}\tFAIL")
            print(f"[{name}] ошибка формата вывода: {err}", file=sys.stderr)
            all_ok = False
            continue

        ok, why, value = evaluate(n, v, cap, demand, x, y, claimed, routes)
        if not ok:
            print(f"{name}\tFAIL")
            print(f"[{name}] {why}", file=sys.stderr)
            all_ok = False
            continue

        print(f"{name}\t{value:.6f}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
