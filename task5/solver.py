#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

SIX_TESTS = (
    "fl_25_2",
    "fl_100_1",
    "fl_200_7",
    "fl_500_7",
    "fl_1000_2",
    "fl_2000_2",
)


def read_instance(lines: Sequence[str]):
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


def dist_mat(n: int, m: int, fx, fy, cx, cy):
    dm = [[0.0] * n for _ in range(m)]
    for i in range(m):
        row = dm[i]
        xi, yi = cx[i], cy[i]
        for j in range(n):
            row[j] = math.hypot(xi - fx[j], yi - fy[j])
    return dm


def k_nearest_per_customer(n: int, m: int, dm, k: int):
    k = min(k, n)
    out = []
    for i in range(m):
        idx = list(range(n))
        idx.sort(key=lambda j: dm[i][j])
        out.append(idx[:k])
    return out


def greedy_assign(n, m, cap, dm, open_mask, order, dem):
    a = [-1] * m
    load = [0.0] * n
    for i in order:
        di = dem[i]
        best_j = -1
        best_dd = 1e300
        row = dm[i]
        for j in range(n):
            if not open_mask[j]:
                continue
            if load[j] + di > cap[j] + 1e-9:
                continue
            dd = row[j]
            if dd < best_dd:
                best_dd = dd
                best_j = j
        if best_j < 0:
            return None
        a[i] = best_j
        load[best_j] += di
    return a, load


def multitry_greedy(n, m, cap, dm, open_mask, dem, tries, rng, reg_order=None):
    order_heavy = sorted(range(m), key=lambda i: -dem[i])
    best = None
    for t in range(tries):
        if t == 0:
            order = order_heavy
        elif t == 1 and reg_order is not None:
            order = reg_order
        else:
            order = list(range(m))
            rng.shuffle(order)
            prefix = order_heavy[: max(1, m // 5)]
            seen = set(prefix)
            rest = [i for i in order if i not in seen]
            order = prefix + rest
        r = greedy_assign(n, m, cap, dm, open_mask, order, dem)
        if r is not None:
            best = r
            break
    return best


def objective(n, s, dm, a):
    used = set(a)
    fix = sum(s[j] for j in used)
    trans = sum(dm[i][a[i]] for i in range(len(a)))
    return fix + trans


def improve_single_moves(n, m, cap, dm, a, load, open_mask, dem, near):
    improved = False
    for i in range(m):
        cur = a[i]
        di = dem[i]
        row = dm[i]
        for j in near[i]:
            if j == cur or not open_mask[j]:
                continue
            if load[j] + di > cap[j] + 1e-9:
                continue
            delta = row[j] - row[cur]
            if delta < -1e-12:
                load[cur] -= di
                load[j] += di
                a[i] = j
                improved = True
                if load[cur] < 1e-9:
                    open_mask[cur] = False
                open_mask[j] = True
                cur = j
    return improved


def build_cheap_mask(n, cap, s, dem):
    need = sum(dem)
    order = sorted(range(n), key=lambda j: (s[j] / max(cap[j], 1e-6), s[j]))
    mask = [False] * n
    got = 0.0
    for j in order:
        mask[j] = True
        got += cap[j]
        if got >= need * 1.02:
            break
    return mask


def regret_order(n, m, dm, dem):
    order = []
    for i in range(m):
        row = dm[i]
        v = sorted(row)
        reg = v[1] - v[0] if n > 1 else 0.0
        order.append((reg, dem[i], i))
    order.sort(reverse=True)
    return [t[2] for t in order]


def improve_single_moves_full(n, m, cap, dm, a, load, open_mask, dem):
    improved = False
    for i in range(m):
        cur = a[i]
        di = dem[i]
        row = dm[i]
        for j in range(n):
            if j == cur or not open_mask[j]:
                continue
            if load[j] + di > cap[j] + 1e-9:
                continue
            delta = row[j] - row[cur]
            if delta < -1e-12:
                load[cur] -= di
                load[j] += di
                a[i] = j
                improved = True
                if load[cur] < 1e-9:
                    open_mask[cur] = False
                open_mask[j] = True
                cur = j
    return improved


def improve_pair_swaps(n, m, cap, dm, a, load, open_mask, dem, rng, attempts):
    improved = False
    for _ in range(attempts):
        i = rng.randrange(m)
        k = rng.randrange(m)
        if i == k:
            continue
        fi, fk = a[i], a[k]
        if fi == fk:
            continue
        di, dk = dem[i], dem[k]
        if load[fi] - di + dk > cap[fi] + 1e-9:
            continue
        if load[fk] - dk + di > cap[fk] + 1e-9:
            continue
        old = dm[i][fi] + dm[k][fk]
        new = dm[i][fk] + dm[k][fi]
        if new < old - 1e-12:
            load[fi] += -di + dk
            load[fk] += -dk + di
            a[i], a[k] = fk, fi
            improved = True
            if load[fi] < 1e-9:
                open_mask[fi] = False
            if load[fk] < 1e-9:
                open_mask[fk] = False
            open_mask[fi] = load[fi] > 1e-9
            open_mask[fk] = load[fk] > 1e-9
    return improved


def try_open_facility(n, m, s, cap, dm, dem, a, load, open_mask, f, rng, tries):
    if open_mask[f]:
        return False
    open_mask[f] = True
    r = multitry_greedy(n, m, cap, dm, open_mask, dem, tries, rng)
    if r is None:
        open_mask[f] = False
        return False
    a_new, load_new = r
    if objective(n, s, dm, a_new) < objective(n, s, dm, a) - 1e-9:
        for i in range(m):
            a[i] = a_new[i]
        for j in range(n):
            load[j] = load_new[j]
        prune_unused(n, load, open_mask)
        return True
    open_mask[f] = False
    return False


def try_close_facility(n, m, s, cap, dm, dem, a, load, open_mask, f, rng, tries):
    if not open_mask[f] or load[f] < 1e-9:
        return False
    open_mask[f] = False
    r = multitry_greedy(n, m, cap, dm, open_mask, dem, tries, rng)
    if r is None:
        open_mask[f] = True
        return False
    a_new, load_new = r
    old_o = objective(n, s, dm, a)
    new_o = objective(n, s, dm, a_new)
    if new_o < old_o - 1e-9:
        for i in range(m):
            a[i] = a_new[i]
        for j in range(n):
            load[j] = load_new[j]
        return True
    open_mask[f] = True
    return False


def prune_unused(n, load, open_mask):
    for j in range(n):
        open_mask[j] = load[j] > 1e-9


def ensure_feasible_init(n, m, s, cap, dm, dem, rng, cheap, reg_order):
    if not cheap:
        open_mask = [True] * n
        tries0 = max(14, min(80, m // 8))
        r = multitry_greedy(n, m, cap, dm, open_mask, dem, tries0, rng, reg_order)
        if r is not None:
            return r, open_mask
    open_mask = [True] * n
    r = multitry_greedy(n, m, cap, dm, open_mask, dem, 140, rng, reg_order)
    if r is None:
        return None, None
    mask = build_cheap_mask(n, cap, s, dem)
    rr = multitry_greedy(n, m, cap, dm, mask, dem, max(24, m // 6), rng, reg_order)
    if rr is not None and objective(n, s, dm, rr[0]) < objective(n, s, dm, r[0]):
        return rr, mask
    open_mask = [True] * n
    return r, open_mask


def solve(
    n,
    m,
    s,
    cap,
    dem,
    dm,
    near,
    deadline,
    seed,
):
    rng = random.Random(seed)
    reg_order = (
        regret_order(n, m, dm, dem) if n * m <= 5_000_000 else None
    )
    r, om = ensure_feasible_init(
        n, m, s, cap, dm, dem, rng, cheap=(rng.random() < 0.35), reg_order=reg_order
    )
    if r is None:
        open_mask = [True] * n
        r = multitry_greedy(n, m, cap, dm, open_mask, dem, 200, rng, reg_order)
        a, load = r
        open_mask = [True] * n
    else:
        a, load = r
        open_mask = om[:] if om else [True] * n
        prune_unused(n, load, open_mask)
    prune_unused(n, load, open_mask)

    tries_close = max(8, min(28, m // 25 + 5))

    pair_budget = max(200, m * 4)

    for _ in range(18):
        if time.perf_counter() >= deadline:
            break
        ch = False
        while time.perf_counter() < deadline and improve_single_moves(
            n, m, cap, dm, a, load, open_mask, dem, near
        ):
            ch = True
        if time.perf_counter() < deadline:
            if improve_pair_swaps(
                n, m, cap, dm, a, load, open_mask, dem, rng, min(pair_budget, 8000)
            ):
                ch = True
        prune_unused(n, load, open_mask)

        order_f = sorted(range(n), key=lambda j: -(s[j] / max(load[j], 1e-6)))
        moved = False
        limit_f = min(n, 120 + n // 20)
        for f in order_f[:limit_f]:
            if time.perf_counter() >= deadline:
                break
            if try_close_facility(n, m, s, cap, dm, dem, a, load, open_mask, f, rng, tries_close):
                moved = True
                prune_unused(n, load, open_mask)
        if not moved and not ch:
            break

    while time.perf_counter() < deadline and improve_single_moves(
        n, m, cap, dm, a, load, open_mask, dem, near
    ):
        pass
    prune_unused(n, load, open_mask)

    while time.perf_counter() < deadline:
        f = rng.randrange(n)
        if open_mask[f]:
            try_close_facility(
                n, m, s, cap, dm, dem, a, load, open_mask, f, rng, min(20, tries_close)
            )
        else:
            try_open_facility(n, m, s, cap, dm, dem, a, load, open_mask, f, rng, min(18, tries_close))
        if time.perf_counter() >= deadline:
            break

    prune_unused(n, load, open_mask)
    while improve_single_moves(n, m, cap, dm, a, load, open_mask, dem, near):
        pass
    if time.perf_counter() < deadline:
        while improve_pair_swaps(
            n, m, cap, dm, a, load, open_mask, dem, rng, min(12000, pair_budget * 2)
        ):
            pass
        while improve_single_moves(n, m, cap, dm, a, load, open_mask, dem, near):
            pass
    prune_unused(n, load, open_mask)
    if n * m < 3_500_000 and time.perf_counter() < deadline:
        for _ in range(3):
            if not improve_single_moves_full(n, m, cap, dm, a, load, open_mask, dem):
                break
            prune_unused(n, load, open_mask)

    return a, objective(n, s, dm, a)


def run_one_instance(data: Sequence[str]) -> None:
    n, m, s, cap, fx, fy, dem, cx, cy = read_instance(data)
    dm = dist_mat(n, m, fx, fy, cx, cy)
    kn = 55 if n * m < 500_000 else 45
    near = k_nearest_per_customer(n, m, dm, kn)

    lim = float(os.environ.get("FL_TIME", "565"))

    best_a = None
    best_v = 1e300
    seeds = (0, 1, 2, 3, 4, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    t_start = time.perf_counter()
    for k, sd in enumerate(seeds):
        used = time.perf_counter() - t_start
        left = lim - used
        if left < 0.5:
            break
        rest = len(seeds) - k
        chunk = left / rest
        deadline = time.perf_counter() + chunk
        a, v = solve(n, m, s, cap, dem, dm, near, deadline, sd + n * 17 + m * 31)
        if v < best_v:
            best_v, best_a = v, a
    out_assign = " ".join(str(best_a[i] + 1) for i in range(m))
    print(f"{best_v:.10f}")
    print(out_assign)


def main():
    if sys.stdin.isatty():
        root = Path(__file__).resolve().parent
        data_dir = root / "data 2"
        for name in SIX_TESTS:
            path = data_dir / name
            print(name, flush=True)
            if not path.is_file():
                print(f"нет файла {path}", file=sys.stderr, flush=True)
                continue
            lines = path.read_text().strip().splitlines()
            run_one_instance(lines)
            print(flush=True)
        return

    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    run_one_instance(data)


if __name__ == "__main__":
    main()
