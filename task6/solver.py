#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass


@dataclass
class Node:
    demand: int
    x: float
    y: float


@dataclass
class Instance:
    n: int
    vehicles: int
    capacity: int
    nodes: list[Node]


def parse_instance(text: str) -> Instance:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    n, vehicles, capacity = map(int, lines[0].split())
    nodes: list[Node] = []
    for i in range(1, n + 1):
        d, x, y = lines[i].split()
        nodes.append(Node(int(d), float(x), float(y)))
    return Instance(n=n, vehicles=vehicles, capacity=capacity, nodes=nodes)


def build_dist(inst: Instance) -> list[list[float]]:
    n = inst.n
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = inst.nodes[i].x
        yi = inst.nodes[i].y
        for j in range(i + 1, n):
            v = math.hypot(xi - inst.nodes[j].x, yi - inst.nodes[j].y)
            d[i][j] = v
            d[j][i] = v
    return d


def route_load(route: list[int], inst: Instance) -> int:
    return sum(inst.nodes[c].demand for c in route)


def route_cost(route: list[int], dist: list[list[float]]) -> float:
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][0]
    return total


def objective(routes: list[list[int]], dist: list[list[float]]) -> float:
    return sum(route_cost(r, dist) for r in routes)


def sweep_construct(inst: Instance, dist: list[list[float]], seed: int) -> tuple[list[list[int]], list[int]]:
    rng = random.Random(seed)
    depot = inst.nodes[0]
    customers = list(range(1, inst.n))

    items: list[tuple[float, int, float]] = []
    for c in customers:
        node = inst.nodes[c]
        angle = math.atan2(node.y - depot.y, node.x - depot.x)
        radius = math.hypot(node.x - depot.x, node.y - depot.y)
        items.append((angle, c, radius))

    shift = rng.random() * 2.0 * math.pi - math.pi
    direction = -1 if rng.random() < 0.5 else 1
    transformed: list[tuple[float, int, float]] = []
    for angle, c, radius in items:
        a = direction * angle + shift
        while a < -math.pi:
            a += 2.0 * math.pi
        while a >= math.pi:
            a -= 2.0 * math.pi
        transformed.append((a, c, radius))
    transformed.sort(key=lambda t: (t[0], t[2]))

    routes: list[list[int]] = []
    loads: list[int] = []
    cur: list[int] = []
    cur_load = 0
    for _, c, _ in transformed:
        demand = inst.nodes[c].demand
        if cur and cur_load + demand > inst.capacity:
            routes.append(cur)
            loads.append(cur_load)
            cur = []
            cur_load = 0
        cur.append(c)
        cur_load += demand
    if cur:
        routes.append(cur)
        loads.append(cur_load)

    reduce_routes_to_vehicle_limit(routes, loads, inst, dist)
    return routes, loads


def sweep_dp_construct(inst: Instance, dist: list[list[float]], seed: int) -> tuple[list[list[int]], list[int]] | None:
    rng = random.Random(seed)
    depot = inst.nodes[0]
    customers = list(range(1, inst.n))
    customers.sort(
        key=lambda c: (
            math.atan2(inst.nodes[c].y - depot.y, inst.nodes[c].x - depot.x),
            math.hypot(inst.nodes[c].x - depot.x, inst.nodes[c].y - depot.y),
        )
    )
    if not customers:
        return [[] for _ in range(inst.vehicles)], [0] * inst.vehicles

    shift = rng.randrange(len(customers))
    order = customers[shift:] + customers[:shift]
    if rng.random() < 0.5:
        order.reverse()

    m = len(order)
    inf = float("inf")
    seg_cost = [[inf] * (m + 1) for _ in range(m + 1)]
    feasible = [[False] * (m + 1) for _ in range(m + 1)]

    for i in range(m):
        load = 0
        path = 0.0
        prev = 0
        for j in range(i, m):
            c = order[j]
            load += inst.nodes[c].demand
            if load > inst.capacity:
                break
            if j == i:
                path = dist[0][c] + dist[c][0]
            else:
                path += dist[prev][c] + dist[c][0] - dist[prev][0]
            prev = c
            feasible[i][j + 1] = True
            seg_cost[i][j + 1] = path

    dp = [[inf] * (m + 1) for _ in range(inst.vehicles + 1)]
    parent = [[-1] * (m + 1) for _ in range(inst.vehicles + 1)]
    dp[0][0] = 0.0

    for k in range(1, inst.vehicles + 1):
        for j in range(1, m + 1):
            best = inf
            best_i = -1
            for i in range(0, j):
                if not feasible[i][j]:
                    continue
                cand = dp[k - 1][i] + seg_cost[i][j]
                if cand < best:
                    best = cand
                    best_i = i
            dp[k][j] = best
            parent[k][j] = best_i

    best_k = -1
    best_value = inf
    for k in range(1, inst.vehicles + 1):
        if dp[k][m] < best_value:
            best_value = dp[k][m]
            best_k = k
    if best_k < 0 or best_value >= inf / 2:
        return None

    cuts: list[tuple[int, int]] = []
    k = best_k
    j = m
    while k > 0 and j > 0:
        i = parent[k][j]
        if i < 0:
            return None
        cuts.append((i, j))
        j = i
        k -= 1
    cuts.reverse()

    routes = [order[i:j] for i, j in cuts]
    loads = [route_load(r, inst) for r in routes]
    return routes, loads


def split_order_dp(order: list[int], inst: Instance, dist: list[list[float]]) -> tuple[list[list[int]], list[int]] | None:
    m = len(order)
    inf = float("inf")
    seg_cost = [[inf] * (m + 1) for _ in range(m + 1)]
    feasible = [[False] * (m + 1) for _ in range(m + 1)]

    for i in range(m):
        load = 0
        path = 0.0
        prev = 0
        for j in range(i, m):
            c = order[j]
            load += inst.nodes[c].demand
            if load > inst.capacity:
                break
            if j == i:
                path = dist[0][c] + dist[c][0]
            else:
                path += dist[prev][c] + dist[c][0] - dist[prev][0]
            prev = c
            feasible[i][j + 1] = True
            seg_cost[i][j + 1] = path

    dp = [[inf] * (m + 1) for _ in range(inst.vehicles + 1)]
    parent = [[-1] * (m + 1) for _ in range(inst.vehicles + 1)]
    dp[0][0] = 0.0
    for k in range(1, inst.vehicles + 1):
        for j in range(1, m + 1):
            best = inf
            best_i = -1
            for i in range(0, j):
                if not feasible[i][j]:
                    continue
                cand = dp[k - 1][i] + seg_cost[i][j]
                if cand < best:
                    best = cand
                    best_i = i
            dp[k][j] = best
            parent[k][j] = best_i

    best_k = -1
    best_val = inf
    for k in range(1, inst.vehicles + 1):
        if dp[k][m] < best_val:
            best_val = dp[k][m]
            best_k = k
    if best_k < 0 or best_val >= inf / 2:
        return None

    cuts: list[tuple[int, int]] = []
    k = best_k
    j = m
    while k > 0 and j > 0:
        i = parent[k][j]
        if i < 0:
            return None
        cuts.append((i, j))
        j = i
        k -= 1
    cuts.reverse()
    routes = [order[i:j] for i, j in cuts]
    loads = [route_load(r, inst) for r in routes]
    return routes, loads


def global_tour_construct(inst: Instance, dist: list[list[float]], seed: int) -> tuple[list[list[int]], list[int]] | None:
    rng = random.Random(seed)
    customers = list(range(1, inst.n))
    if not customers:
        return [[] for _ in range(inst.vehicles)], [0] * inst.vehicles

    start = rng.choice(customers)
    unvisited = set(customers)
    order = [start]
    unvisited.remove(start)
    while unvisited:
        cur = order[-1]
        nxt = min(unvisited, key=lambda c: dist[cur][c])
        order.append(nxt)
        unvisited.remove(nxt)

    improved = True
    while improved:
        improved = False
        m = len(order)
        for i in range(m - 1):
            a = order[i]
            b = order[(i + 1) % m]
            for k in range(i + 2, m):
                if k == m - 1 and i == 0:
                    continue
                c = order[k]
                d = order[(k + 1) % m]
                old = dist[a][b] + dist[c][d]
                new = dist[a][c] + dist[b][d]
                if new + 1e-12 < old:
                    order[i + 1 : k + 1] = reversed(order[i + 1 : k + 1])
                    improved = True
                    break
            if improved:
                break

    return split_order_dp(order, inst, dist)


def savings_construct(inst: Instance, dist: list[list[float]], seed: int) -> tuple[list[list[int]], list[int]]:
    rng = random.Random(seed)
    customers = list(range(1, inst.n))

    routes: list[list[int]] = [[c] for c in customers]
    loads: list[int] = [inst.nodes[c].demand for c in customers]
    route_of = {c: idx for idx, c in enumerate(customers)}

    savings: list[tuple[float, int, int]] = []
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            s = dist[0][i] + dist[0][j] - dist[i][j]
            s += 1e-6 * rng.random()
            savings.append((s, i, j))
    savings.sort(reverse=True)

    for _, i, j in savings:
        ri = route_of.get(i, -1)
        rj = route_of.get(j, -1)
        if ri < 0 or rj < 0 or ri == rj:
            continue
        if loads[ri] + loads[rj] > inst.capacity:
            continue
        a = routes[ri]
        b = routes[rj]
        if not a or not b:
            continue

        merged: list[int] | None = None
        if a[-1] == i and b[0] == j:
            merged = a + b
        elif a[0] == i and b[-1] == j:
            merged = b + a
        elif a[0] == i and b[0] == j:
            merged = list(reversed(a)) + b
        elif a[-1] == i and b[-1] == j:
            merged = a + list(reversed(b))
        if merged is None:
            continue

        new_id = ri
        old_id = rj
        routes[new_id] = merged
        loads[new_id] += loads[old_id]
        routes[old_id] = []
        loads[old_id] = 0

        for c in merged:
            route_of[c] = new_id

    routes = [r for r in routes if r]
    loads = [route_load(r, inst) for r in routes]
    reduce_routes_to_vehicle_limit(routes, loads, inst, dist)
    return routes, loads


def best_insert_position(route: list[int], customer: int, dist: list[list[float]]) -> tuple[float, int]:
    if not route:
        return dist[0][customer] + dist[customer][0], 0
    best_delta = float("inf")
    best_pos = 0

    delta = dist[0][customer] + dist[customer][route[0]] - dist[0][route[0]]
    if delta < best_delta:
        best_delta = delta
        best_pos = 0

    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        delta = dist[a][customer] + dist[customer][b] - dist[a][b]
        if delta < best_delta:
            best_delta = delta
            best_pos = i + 1

    delta = dist[route[-1]][customer] + dist[customer][0] - dist[route[-1]][0]
    if delta < best_delta:
        best_delta = delta
        best_pos = len(route)
    return best_delta, best_pos


def reduce_routes_to_vehicle_limit(
    routes: list[list[int]], loads: list[int], inst: Instance, dist: list[list[float]]
) -> None:
    while len(routes) > inst.vehicles:
        idx_small = min(range(len(routes)), key=lambda i: (loads[i], len(routes[i])))
        route = routes[idx_small]
        if not route:
            del routes[idx_small]
            del loads[idx_small]
            continue

        moved_all = True
        for c in list(route):
            demand = inst.nodes[c].demand
            best = None
            for j in range(len(routes)):
                if j == idx_small:
                    continue
                if loads[j] + demand > inst.capacity:
                    continue
                delta, pos = best_insert_position(routes[j], c, dist)
                if best is None or delta < best[0]:
                    best = (delta, j, pos)
            if best is None:
                moved_all = False
                break
            _, j, pos = best
            routes[j].insert(pos, c)
            loads[j] += demand
            route.remove(c)

        if moved_all and not route:
            del routes[idx_small]
            del loads[idx_small]
        else:
            break


def two_opt(route: list[int], dist: list[list[float]]) -> bool:
    n = len(route)
    if n < 4:
        return False
    for i in range(n - 2):
        a_prev = 0 if i == 0 else route[i - 1]
        a = route[i]
        for k in range(i + 1, n - 1):
            b = route[k]
            b_next = route[k + 1]
            old = dist[a_prev][a] + dist[b][b_next]
            new = dist[a_prev][b] + dist[a][b_next]
            if new + 1e-12 < old:
                route[i : k + 1] = reversed(route[i : k + 1])
                return True

        b = route[-1]
        old = dist[a_prev][a] + dist[b][0]
        new = dist[a_prev][b] + dist[a][0]
        if new + 1e-12 < old:
            route[i:] = reversed(route[i:])
            return True
    return False


def improve_intra_routes(routes: list[list[int]], dist: list[list[float]]) -> bool:
    changed = False
    for r in routes:
        while two_opt(r, dist):
            changed = True
    return changed


def relocate_between_routes(
    routes: list[list[int]], loads: list[int], inst: Instance, dist: list[list[float]]
) -> bool:
    for a in range(len(routes)):
        ra = routes[a]
        if not ra:
            continue
        for ia, customer in enumerate(list(ra)):
            demand = inst.nodes[customer].demand
            prev_a = 0 if ia == 0 else ra[ia - 1]
            next_a = 0 if ia == len(ra) - 1 else ra[ia + 1]
            remove_delta = dist[prev_a][next_a] - dist[prev_a][customer] - dist[customer][next_a]

            for b in range(len(routes)):
                if a == b:
                    continue
                if loads[b] + demand > inst.capacity:
                    continue
                rb = routes[b]
                insert_delta, pos = best_insert_position(rb, customer, dist)
                total_delta = remove_delta + insert_delta
                if total_delta < -1e-9:
                    del routes[a][ia]
                    routes[b].insert(pos, customer)
                    loads[a] -= demand
                    loads[b] += demand
                    if not routes[a]:
                        del routes[a]
                        del loads[a]
                    return True
    return False


def swap_between_routes(
    routes: list[list[int]],
    loads: list[int],
    inst: Instance,
    dist: list[list[float]],
    rng: random.Random,
    attempts: int,
) -> bool:
    if len(routes) < 2:
        return False
    for _ in range(attempts):
        a = rng.randrange(len(routes))
        b = rng.randrange(len(routes))
        if a == b or not routes[a] or not routes[b]:
            continue
        ia = rng.randrange(len(routes[a]))
        ib = rng.randrange(len(routes[b]))
        ca = routes[a][ia]
        cb = routes[b][ib]
        da = inst.nodes[ca].demand
        db = inst.nodes[cb].demand
        new_la = loads[a] - da + db
        new_lb = loads[b] - db + da
        if new_la > inst.capacity or new_lb > inst.capacity:
            continue

        ra = routes[a]
        rb = routes[b]
        pa = 0 if ia == 0 else ra[ia - 1]
        na = 0 if ia == len(ra) - 1 else ra[ia + 1]
        pb = 0 if ib == 0 else rb[ib - 1]
        nb = 0 if ib == len(rb) - 1 else rb[ib + 1]

        old_a = dist[pa][ca] + dist[ca][na]
        old_b = dist[pb][cb] + dist[cb][nb]
        new_a = dist[pa][cb] + dist[cb][na]
        new_b = dist[pb][ca] + dist[ca][nb]
        if ia > 0 and ia < len(ra) - 1 and pa == na:
            pass
        if ib > 0 and ib < len(rb) - 1 and pb == nb:
            pass

        delta = (new_a + new_b) - (old_a + old_b)
        if delta < -1e-9:
            routes[a][ia], routes[b][ib] = routes[b][ib], routes[a][ia]
            loads[a], loads[b] = new_la, new_lb
            return True
    return False


def repack_exact_vehicle_count(
    routes: list[list[int]], inst: Instance, dist: list[list[float]], seed: int
) -> list[list[int]] | None:
    customers: list[int] = []
    for r in routes:
        customers.extend(r)

    if len(customers) != inst.n - 1:
        return None

    rng = random.Random(seed)
    tries = 20
    for _ in range(tries):
        order = customers[:]
        order.sort(key=lambda c: inst.nodes[c].demand, reverse=True)

        for i in range(len(order) - 1):
            if inst.nodes[order[i]].demand == inst.nodes[order[i + 1]].demand and rng.random() < 0.5:
                order[i], order[i + 1] = order[i + 1], order[i]

        packed = [[] for _ in range(inst.vehicles)]
        loads = [0] * inst.vehicles
        ok = True
        for c in order:
            demand = inst.nodes[c].demand
            candidates: list[tuple[float, int, int]] = []
            for k in range(inst.vehicles):
                if loads[k] + demand > inst.capacity:
                    continue
                delta, pos = best_insert_position(packed[k], c, dist)
                candidates.append((delta, k, pos))
            if not candidates:
                ok = False
                break
            candidates.sort(key=lambda t: t[0])
            pick = candidates[min(len(candidates) - 1, rng.randrange(min(3, len(candidates))))]
            _, k, pos = pick
            packed[k].insert(pos, c)
            loads[k] += demand

        if ok:
            return packed
    return None


def complete_vehicle_count(routes: list[list[int]], inst: Instance) -> list[list[int]]:
    while len(routes) < inst.vehicles:
        routes.append([])
    return routes


def validate_solution(routes: list[list[int]], inst: Instance) -> bool:
    seen = [0] * inst.n
    for r in routes:
        load = 0
        for c in r:
            if c <= 0 or c >= inst.n:
                return False
            seen[c] += 1
            load += inst.nodes[c].demand
        if load > inst.capacity:
            return False
    for c in range(1, inst.n):
        if seen[c] != 1:
            return False
    return True


def local_search(
    routes: list[list[int]],
    loads: list[int],
    inst: Instance,
    dist: list[list[float]],
    deadline: float,
    seed: int,
) -> tuple[list[list[int]], list[int]]:
    rng = random.Random(seed)

    improve_intra_routes(routes, dist)
    while time.perf_counter() < deadline:
        changed = False
        if relocate_between_routes(routes, loads, inst, dist):
            changed = True
            improve_intra_routes(routes, dist)
        if time.perf_counter() >= deadline:
            break
        if swap_between_routes(routes, loads, inst, dist, rng, attempts=600):
            changed = True
            improve_intra_routes(routes, dist)
        if not changed:
            break
    return routes, loads


def solve(inst: Instance, time_limit: float) -> tuple[float, list[list[int]]]:
    dist = build_dist(inst)
    start = time.perf_counter()

    best_cost = float("inf")
    best_routes: list[list[int]] | None = None
    seeds = [11, 29, 43, 71, 97, 131, 193, 257, 313, 401, 557, 701, 809, 919, 1031, 1237]

    for idx, seed in enumerate(seeds):
        now = time.perf_counter()
        if now - start > time_limit - 0.1:
            break
        remaining = time_limit - (now - start)
        chunk = max(0.5, remaining / (len(seeds) - idx))
        deadline = time.perf_counter() + chunk

        mode = idx % 4
        if mode == 0:
            routes, loads = savings_construct(inst, dist, seed)
        elif mode == 1:
            routes, loads = sweep_construct(inst, dist, seed)
        elif mode == 2:
            constructed = sweep_dp_construct(inst, dist, seed)
            if constructed is None:
                continue
            routes, loads = constructed
        else:
            constructed = global_tour_construct(inst, dist, seed)
            if constructed is None:
                continue
            routes, loads = constructed

        if len(routes) > inst.vehicles:
            packed = repack_exact_vehicle_count(routes, inst, dist, seed + 12345)
            if packed is None:
                continue
            routes = packed

        routes = complete_vehicle_count(routes, inst)
        loads = [route_load(r, inst) for r in routes]
        if not validate_solution(routes, inst):
            continue

        routes, loads = local_search(routes, loads, inst, dist, deadline, seed + 777)
        routes = complete_vehicle_count(routes, inst)
        if not validate_solution(routes, inst):
            continue
        cost = objective(routes, dist)
        if cost < best_cost:
            best_cost = cost
            best_routes = [r[:] for r in routes]

    if best_routes is None:

        routes: list[list[int]] = []
        loads: list[int] = []
        cur: list[int] = []
        cur_load = 0
        for c in range(1, inst.n):
            d = inst.nodes[c].demand
            if cur and cur_load + d > inst.capacity:
                routes.append(cur)
                loads.append(cur_load)
                cur = []
                cur_load = 0
            cur.append(c)
            cur_load += d
        if cur:
            routes.append(cur)
            loads.append(cur_load)
        reduce_routes_to_vehicle_limit(routes, loads, inst, dist)
        if len(routes) > inst.vehicles:
            packed = repack_exact_vehicle_count(routes, inst, dist, 777)
            if packed is not None:
                routes = packed
        routes = complete_vehicle_count(routes, inst)
        best_routes = routes
        best_cost = objective(best_routes, dist)

    return best_cost, best_routes


def format_output(value: float, routes: list[list[int]]) -> str:
    lines = [f"{value:.6f} 0"]
    for r in routes:
        if r:
            lines.append("0 " + " ".join(map(str, r)) + " 0")
        else:
            lines.append("0 0")
    return "\n".join(lines) + "\n"


def main() -> None:
    text = sys.stdin.read()
    if not text.strip():
        return
    inst = parse_instance(text)
    time_limit = float(os.environ.get("VRP_TIME", "560"))
    value, routes = solve(inst, time_limit=time_limit)
    sys.stdout.write(format_output(value, routes))


if __name__ == "__main__":
    main()
