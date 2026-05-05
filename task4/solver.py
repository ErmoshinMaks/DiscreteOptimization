#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import sys
from array import array
from dataclasses import dataclass


EPS = 1e-9
NEAREST_K = 120
RNG_SEED = 42
OPEN_COST_FACTOR = 1.2


@dataclass
class Instance:
    n: int
    m: int
    setup: list[float]
    cap: list[float]
    fx: list[float]
    fy: list[float]
    dem: list[float]
    cx: list[float]
    cy: list[float]


def parse_instance(text: str) -> Instance:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    n, m = map(int, lines[0].split())

    setup = [0.0] * n
    cap = [0.0] * n
    fx = [0.0] * n
    fy = [0.0] * n
    for i in range(n):
        s, c, x, y = map(float, lines[1 + i].split())
        setup[i] = s
        cap[i] = c
        fx[i] = x
        fy[i] = y

    dem = [0.0] * m
    cx = [0.0] * m
    cy = [0.0] * m
    for i in range(m):
        d, x, y = map(float, lines[1 + n + i].split())
        dem[i] = d
        cx[i] = x
        cy[i] = y

    return Instance(n, m, setup, cap, fx, fy, dem, cx, cy)


def build_distance_matrix(inst: Instance) -> list[array]:
    dist: list[array] = []
    for c in range(inst.m):
        row = array("d")
        x = inst.cx[c]
        y = inst.cy[c]
        for f in range(inst.n):
            row.append(math.hypot(x - inst.fx[f], y - inst.fy[f]))
        dist.append(row)
    return dist


def nearest_lists(dist: list[array], k: int) -> list[list[int]]:
    n = len(dist[0]) if dist else 0
    use_k = min(k, n)
    result: list[list[int]] = []
    for row in dist:
        ids = list(range(n))
        ids.sort(key=row.__getitem__)
        result.append(ids[:use_k])
    return result


def greedy_initial_solution(
    inst: Instance,
    dist: list[array],
    rng: random.Random,
) -> tuple[list[int], list[bool], list[float]]:
    remaining = inst.cap[:]
    open_mask = [False] * inst.n
    assign = [-1] * inst.m

    order = list(range(inst.m))
    order.sort(key=lambda c: (-inst.dem[c], rng.random()))

    for c in order:
        best_j = -1
        best_cost = float("inf")
        for j in range(inst.n):
            if remaining[j] + EPS < inst.dem[c]:
                continue
            cand = dist[c][j]
            if not open_mask[j]:
                cand += OPEN_COST_FACTOR * inst.setup[j] * (inst.dem[c] / max(inst.cap[j], EPS))
            if cand < best_cost:
                best_cost = cand
                best_j = j
        if best_j < 0:
            raise ValueError("No feasible facility for customer")
        assign[c] = best_j
        open_mask[best_j] = True
        remaining[best_j] -= inst.dem[c]

    return assign, open_mask, remaining


def reassign_to_open_facilities(
    inst: Instance,
    dist: list[array],
    near: list[list[int]],
    assign: list[int],
    open_mask: list[bool],
    remaining: list[float],
    rng: random.Random,
) -> bool:
    moved = False
    order = list(range(inst.m))
    rng.shuffle(order)

    for c in order:
        cur = assign[c]
        best = cur
        best_delta = 0.0
        cur_d = dist[c][cur]
        for j in near[c]:
            if j == cur or (not open_mask[j]):
                continue
            if remaining[j] + EPS < inst.dem[c]:
                continue
            delta = dist[c][j] - cur_d
            if delta < best_delta:
                best_delta = delta
                best = j
        if best != cur:
            remaining[cur] += inst.dem[c]
            remaining[best] -= inst.dem[c]
            assign[c] = best
            moved = True
    return moved


def try_open_one_facility(
    inst: Instance,
    dist: list[array],
    assign: list[int],
    open_mask: list[bool],
    remaining: list[float],
    rng: random.Random,
) -> bool:
    closed = [j for j in range(inst.n) if not open_mask[j]]
    if not closed:
        return False

    rng.shuffle(closed)
    candidates = closed[: min(240, len(closed))]

    current_d = [dist[c][assign[c]] for c in range(inst.m)]
    best_gain = 0.0
    best_j = -1
    best_selected: list[int] = []

    for j in candidates:
        items: list[tuple[float, float, int]] = []
        for c in range(inst.m):
            save = current_d[c] - dist[c][j]
            if save > 1e-8:
                items.append((save / max(inst.dem[c], EPS), save, c))
        if not items:
            continue

        items.sort(reverse=True)
        cap_left = inst.cap[j]
        chosen: list[int] = []
        gain = -inst.setup[j]
        for _, save, c in items:
            d = inst.dem[c]
            if d <= cap_left + EPS:
                chosen.append(c)
                cap_left -= d
                gain += save
        if gain > best_gain and chosen:
            best_gain = gain
            best_j = j
            best_selected = chosen

    if best_j < 0:
        return False

    open_mask[best_j] = True
    load_new = 0.0
    for c in best_selected:
        old = assign[c]
        remaining[old] += inst.dem[c]
        assign[c] = best_j
        load_new += inst.dem[c]
    remaining[best_j] = inst.cap[best_j] - load_new
    return True


def try_close_one_facility(
    inst: Instance,
    dist: list[array],
    near: list[list[int]],
    assign: list[int],
    open_mask: list[bool],
    remaining: list[float],
) -> bool:
    open_idx = [j for j in range(inst.n) if open_mask[j]]
    if len(open_idx) <= 1:
        return False

    open_idx.sort(key=lambda j: -inst.setup[j])
    for j in open_idx[:120]:
        customers = [c for c in range(inst.m) if assign[c] == j]
        if not customers:
            open_mask[j] = False
            remaining[j] = inst.cap[j]
            return True

        tmp_remaining = remaining[:]
        moves: dict[int, int] = {}
        extra_distance = 0.0

        customers.sort(key=lambda c: -inst.dem[c])
        feasible = True
        for c in customers:
            best = -1
            best_d = float("inf")
            for k in near[c]:
                if k == j or (not open_mask[k]):
                    continue
                if tmp_remaining[k] + EPS < inst.dem[c]:
                    continue
                d = dist[c][k]
                if d < best_d:
                    best_d = d
                    best = k
            if best < 0:
                feasible = False
                break
            moves[c] = best
            tmp_remaining[best] -= inst.dem[c]
            extra_distance += best_d - dist[c][j]

        if (not feasible) or (extra_distance >= inst.setup[j] - 1e-7):
            continue

        for c, k in moves.items():
            assign[c] = k
        for k in range(inst.n):
            remaining[k] = tmp_remaining[k]
        open_mask[j] = False
        remaining[j] = inst.cap[j]
        return True
    return False


def total_objective(inst: Instance, dist: list[array], assign: list[int], open_mask: list[bool]) -> float:
    service = 0.0
    for c in range(inst.m):
        service += dist[c][assign[c]]
    used = set(assign)
    fixed = 0.0
    for j in used:
        fixed += inst.setup[j]
    return fixed + service


def solve_one_start(
    inst: Instance,
    dist: list[array],
    near: list[list[int]],
    seed: int,
) -> tuple[float, list[int]]:
    rng = random.Random(seed)
    assign, open_mask, remaining = greedy_initial_solution(inst, dist, rng)

    for _ in range(30):
        moved = reassign_to_open_facilities(inst, dist, near, assign, open_mask, remaining, rng)
        opened = try_open_one_facility(inst, dist, assign, open_mask, remaining, rng)
        if opened:
            reassign_to_open_facilities(inst, dist, near, assign, open_mask, remaining, rng)
            reassign_to_open_facilities(inst, dist, near, assign, open_mask, remaining, rng)
        closed = try_close_one_facility(inst, dist, near, assign, open_mask, remaining)
        if not (moved or opened or closed):
            break

    value = total_objective(inst, dist, assign, open_mask)
    return value, assign


def solve(inst: Instance) -> tuple[float, list[int]]:
    dist = build_distance_matrix(inst)
    near = nearest_lists(dist, NEAREST_K)

    if inst.n <= 200:
        starts = 16
    elif inst.n <= 1000:
        starts = 14
    else:
        starts = 8

    best_value = float("inf")
    best_assign: list[int] = []
    for t in range(starts):
        value, assign = solve_one_start(inst, dist, near, RNG_SEED + 9973 * t)
        if value < best_value:
            best_value = value
            best_assign = assign[:]
    return best_value, best_assign


def main() -> None:
    text = sys.stdin.read()
    if not text.strip():
        return
    inst = parse_instance(text)
    value, assign = solve(inst)
    print(f"{value:.6f}")
    print(" ".join(str(a + 1) for a in assign))


if __name__ == "__main__":
    main()
