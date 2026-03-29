#!/usr/bin/env python3

from __future__ import annotations

import random
import sys
import time


def parse_input(text: str) -> tuple[int, list[list[int]]]:
    lines = text.splitlines()
    n, m = map(int, lines[0].split())
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, m + 1):
        a, b = map(int, lines[i].split())
        adj[a].append(b)
        adj[b].append(a)
    return n, adj


def objective(colors: list[int]) -> int:
    return len(set(colors))


def greedy_order(adj: list[list[int]], order: list[int]) -> list[int]:
    n = len(adj)
    colors = [-1] * n
    for v in order:
        used: set[int] = set()
        for u in adj[v]:
            cu = colors[u]
            if cu >= 0:
                used.add(cu)
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors


def welsh_powell(adj: list[list[int]], n: int) -> list[int]:
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    return greedy_order(adj, order)


def dsatur(adj: list[list[int]], n: int) -> list[int]:
    colors = [-1] * n
    degrees = [len(adj[v]) for v in range(n)]
    sat: list[set[int]] = [set() for _ in range(n)]
    uncolored = set(range(n))
    while uncolored:
        v = max(uncolored, key=lambda x: (len(sat[x]), degrees[x]))
        used = sat[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        uncolored.remove(v)
        for u in adj[v]:
            if u in uncolored:
                sat[u].add(c)
    return colors


def smallest_last_order(adj: list[list[int]], n: int) -> list[int]:
    deg = [len(adj[v]) for v in range(n)]
    alive = set(range(n))
    order_rev: list[int] = []
    while alive:
        v = min(alive, key=lambda x: deg[x])
        order_rev.append(v)
        alive.remove(v)
        for u in adj[v]:
            if u in alive:
                deg[u] -= 1
    order_rev.reverse()
    return order_rev


def is_proper(colors: list[int], adj: list[list[int]]) -> bool:
    for v, nbrs in enumerate(adj):
        cv = colors[v]
        for u in nbrs:
            if colors[u] == cv:
                return False
    return True


def local_improve(colors: list[int], adj: list[list[int]], n: int) -> None:
    order = list(range(n))
    changed = True
    while changed:
        changed = False
        random.shuffle(order)
        for v in order:
            neigh = adj[v]
            used: set[int] = set()
            for u in neigh:
                used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            if c < colors[v]:
                colors[v] = c
                changed = True


def ig_remove_recolor(colors: list[int], adj: list[list[int]], n: int, rng: random.Random) -> None:
    classes: dict[int, list[int]] = {}
    for v in range(n):
        classes.setdefault(colors[v], []).append(v)
    cid = rng.choice(list(classes.keys()))
    verts = classes[cid]
    for v in verts:
        colors[v] = -1
    rng.shuffle(verts)
    for v in verts:
        used: set[int] = set()
        for u in adj[v]:
            cu = colors[u]
            if cu >= 0:
                used.add(cu)
        c = 0
        while c in used:
            c += 1
        colors[v] = c


def dsatur_k_init(adj: list[list[int]], n: int, k: int, rng: random.Random) -> list[int]:
    colors = [-1] * n
    degrees = [len(adj[v]) for v in range(n)]
    sat: list[set[int]] = [set() for _ in range(n)]
    uncolored = set(range(n))
    while uncolored:
        v = max(uncolored, key=lambda x: (len(sat[x]), degrees[x]))
        used = sat[v]
        c = 0
        while c < k and c in used:
            c += 1
        if c >= k:
            c = rng.randrange(k)
        colors[v] = c
        uncolored.remove(v)
        for u in adj[v]:
            if u in uncolored:
                sat[u].add(c)
    return colors


def greedy_k_init(adj: list[list[int]], n: int, k: int, rng: random.Random) -> list[int]:
    order = list(range(n))
    rng.shuffle(order)
    order.sort(key=lambda v: len(adj[v]), reverse=True)
    colors = [-1] * n
    for v in order:
        used: set[int] = set()
        for u in adj[v]:
            cu = colors[u]
            if cu >= 0:
                used.add(cu)
        choices = [c for c in range(k) if c not in used]
        colors[v] = rng.choice(choices) if choices else rng.randrange(k)
    return colors


def edge_conflicts(adj: list[list[int]], colors: list[int], n: int) -> int:
    t = 0
    for v in range(n):
        cv = colors[v]
        for u in adj[v]:
            if u > v and colors[u] == cv:
                t += 1
    return t


def vertex_is_conflicting(colors: list[int], adj: list[list[int]], v: int) -> bool:
    cv = colors[v]
    for u in adj[v]:
        if colors[u] == cv:
            return True
    return False


def move_delta_conflict(v: int, new_c: int, colors: list[int], adj: list[list[int]]) -> int:
    old = colors[v]
    if old == new_c:
        return 0
    d = 0
    for u in adj[v]:
        if colors[u] == old:
            d -= 1
        if colors[u] == new_c:
            d += 1
    return d


def search_proper_k_coloring(
    adj: list[list[int]], n: int, k: int, rng: random.Random, deadline: float
) -> list[int] | None:
    if k <= 0:
        return None

    restart_idx = 0
    while time.perf_counter() < deadline:
        r = restart_idx % 4
        if r == 0:
            colors = dsatur_k_init(adj, n, k, rng)
        elif r == 1:
            colors = greedy_k_init(adj, n, k, rng)
        elif r == 2:
            colors = [rng.randrange(k) for _ in range(n)]
        else:
            colors = dsatur_k_init(adj, n, k, rng)
            for _ in range(min(30, n // 20)):
                v = rng.randrange(n)
                colors[v] = rng.randrange(k)
        restart_idx += 1

        tc = edge_conflicts(adj, colors, n)
        step = 0
        idle = 0

        while tc > 0 and time.perf_counter() < deadline:
            v = rng.randrange(n)
            if not vertex_is_conflicting(colors, adj, v):
                idle += 1
                if idle > 200_000:
                    break
                continue
            idle = 0

            old = colors[v]
            best_c = old
            best_d = 10**9
            for c in range(k):
                if c == old:
                    continue
                d = move_delta_conflict(v, c, colors, adj)
                if d < best_d or (d == best_d and rng.random() < 0.5):
                    best_d = d
                    best_c = c

            if best_d > 0:
                if rng.random() < 0.004:
                    best_c = rng.randrange(k)
                    while best_c == old:
                        best_c = rng.randrange(k)
                    best_d = move_delta_conflict(v, best_c, colors, adj)
                else:
                    continue

            colors[v] = best_c
            tc += best_d
            step += 1
            if step > 15_000_000:
                break

        if tc == 0:
            return colors

    return None


def try_k_coloring_backtrack(
    adj: list[list[int]], n: int, k: int, rng: random.Random, max_nodes: int
) -> list[int] | None:
    if k <= 0:
        return None
    order = list(range(n))
    rng.shuffle(order)
    order.sort(key=lambda v: len(adj[v]), reverse=True)
    colors = [-1] * n
    nodes = 0

    def dfs(i: int) -> bool:
        nonlocal nodes
        if nodes >= max_nodes:
            return False
        if i >= n:
            return True
        v = order[i]
        used: set[int] = set()
        for u in adj[v]:
            cu = colors[u]
            if cu >= 0:
                used.add(cu)
        opts = [c for c in range(k) if c not in used]
        rng.shuffle(opts)
        for c in opts:
            nodes += 1
            colors[v] = c
            if dfs(i + 1):
                return True
            colors[v] = -1
        return False

    if dfs(0):
        return colors
    return None


def portfolio_initial(adj: list[list[int]], n: int, rng: random.Random) -> list[int]:
    best = list(range(n))
    best_obj = n

    candidates = [
        welsh_powell(adj, n),
        dsatur(adj, n),
        greedy_order(adj, smallest_last_order(adj, n)),
    ]
    for c in candidates:
        o = objective(c)
        if o < best_obj:
            best, best_obj = c, o

    restarts = max(120, min(6000, 350000 // max(1, n)))
    for _ in range(restarts):
        order = list(range(n))
        rng.shuffle(order)
        c = greedy_order(adj, order)
        o = objective(c)
        if o < best_obj:
            best, best_obj = c[:], o
    return best


def solve_it(input_data: str, time_limit: float) -> str:
    start = time.perf_counter()
    rng = random.Random(0xC010F00D)

    n, adj = parse_input(input_data)
    deadline = start + time_limit
    if n >= 600:
        reserve = min(400.0, max(90.0, time_limit * 0.44))
    else:
        reserve = min(320.0, max(45.0, time_limit * 0.38))
    phase1_deadline = deadline - reserve

    best = portfolio_initial(adj, n, rng)
    best_obj = objective(best)

    if n <= 95 and time_limit > 2:
        low = 1
        high = best_obj
        while low < high:
            mid = (low + high) // 2
            elapsed = time.perf_counter() - start
            if elapsed > phase1_deadline - 0.5:
                break
            rem = int(min(1_200_000, (phase1_deadline - elapsed) * 250_000))
            col = try_k_coloring_backtrack(adj, n, mid, rng, rem)
            if col is not None:
                best = col
                best_obj = mid
                high = mid
            else:
                low = mid + 1

    cur = best[:]
    cur_obj = best_obj

    while time.perf_counter() < phase1_deadline:
        local_improve(cur, adj, n)
        o = objective(cur)
        if o < cur_obj:
            cur_obj = o
            best = cur[:]
            best_obj = o

        ig_remove_recolor(cur, adj, n, rng)
        local_improve(cur, adj, n)
        o = objective(cur)
        if o < cur_obj:
            cur_obj = o
            best = cur[:]
            best_obj = o

        if rng.random() < 0.12:
            cur = best[:]
            cur_obj = best_obj

    while best_obj > 1 and time.perf_counter() < deadline - 0.05:
        k_target = best_obj - 1
        now = time.perf_counter()
        remaining = deadline - now
        chunk = max(18.0, min(remaining * 0.72, remaining - 0.1))
        chunk_end = now + chunk
        col = search_proper_k_coloring(adj, n, k_target, rng, chunk_end)
        if col is not None and is_proper(col, adj):
            best = col
            best_obj = k_target
        elif time.perf_counter() >= deadline - 0.05:
            break

    assert is_proper(best, adj)
    runtime = time.perf_counter() - start
    k = len(set(best))
    line0 = f"{k} 0"
    line1 = " ".join(str(x) for x in best)
    line2 = str(runtime)
    return f"{line0}\n{line1}\n{line2}\n"


def main() -> None:
    default_limit = 580.0
    if len(sys.argv) > 1:
        path = sys.argv[1].strip()
        with open(path, encoding="utf-8") as f:
            data = f.read()
        if len(sys.argv) > 2:
            limit = float(sys.argv[2])
        else:
            limit = default_limit
        print(solve_it(data, limit), end="")
    else:
        data = sys.stdin.read()
        print(solve_it(data, default_limit), end="")


if __name__ == "__main__":
    main()
