#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Literal, Optional, Sequence, Tuple


BytesLine = bytes


@dataclass(frozen=True)
class InstanceInfo:
    n: int
    m: int
    path: str
    delete_after: bool


OutputFormat = Literal["indices", "coursera"]


@dataclass(frozen=True)
class Solution:
    set_ids: List[int]
    objective: int
    m: int


def _parse_ints_from_bytes_line(line: BytesLine) -> List[int]:
    parts = line.split()
    return [int(x) for x in parts]


def _spool_stdin_to_tempfile() -> str:
    tmp = tempfile.NamedTemporaryFile(prefix="setcover_", delete=False)
    try:
        for chunk in sys.stdin.buffer:
            tmp.write(chunk)
    finally:
        tmp.close()
    return tmp.name


def _open_instance(args: argparse.Namespace) -> InstanceInfo:
    if args.input_path is not None:
        return InstanceInfo(n=-1, m=-1, path=args.input_path, delete_after=False)

    tmp_path = _spool_stdin_to_tempfile()
    return InstanceInfo(n=-1, m=-1, path=tmp_path, delete_after=True)


def _read_header(path: str) -> Tuple[int, int]:
    with open(path, "rb") as f:
        header = f.readline()
    if not header:
        raise ValueError("Empty input")
    n, m = _parse_ints_from_bytes_line(header)
    return n, m


def _iter_sets(path: str, m: int) -> Iterator[Tuple[int, int, List[int]]]:
    with open(path, "rb") as f:
        _ = f.readline()
        for set_id in range(m):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading set {set_id}/{m}")
            ints = _parse_ints_from_bytes_line(line)
            if len(ints) < 2:
                raise ValueError(f"Set line must have cost and >=1 element, got: {line!r}")
            cost = ints[0]
            elems = ints[1:]
            yield set_id, cost, elems


def _candidate_limit(n: int, m: int) -> int:
    # L = сколько лучших множеств на элемент мы сохраняем.
    # При большом m уменьшаем L, чтобы память/время были ограничены.
    if m >= 800_000:
        return 18
    if m >= 200_000:
        return 25
    if n >= 10_000:
        return 30
    return 40


def _build_candidates(path: str, n: int, m: int, per_elem_limit: int) -> Tuple[List[int], List[int]]:
    # Кандидаты строятся как объединение топ-L множеств для каждого элемента
    # по метрике cost/|S|. Это резко уменьшает задачу при огромном m.
    freq = [0] * n
    best_per_elem: List[List[Tuple[float, int]]] = [[] for _ in range(n)]

    for set_id, cost, elems in _iter_sets(path, m):
        k = len(elems)
        if k <= 0:
            continue
        score = cost / k  # "цена за элемент" (грубая, но быстрая метрика)
        neg_score = -score
        for e in elems:
            if e < 0 or e >= n:
                raise ValueError(f"Element index out of range: {e} for n={n}")
            freq[e] += 1
            h = best_per_elem[e]
            heapq.heappush(h, (neg_score, set_id))
            if len(h) > per_elem_limit:
                heapq.heappop(h)

    candidate_ids = set()
    for h in best_per_elem:
        for _, set_id in h:
            candidate_ids.add(set_id)

    return freq, sorted(candidate_ids)


@dataclass
class CandidateSets:
    set_ids: List[int] 
    costs: List[int]
    elems: List[List[int]]
    inv: Optional[List[List[int]]]
    best_local_for_elem: List[int]


def _load_candidate_sets(path: str, n: int, m: int, candidate_set_ids: Sequence[int]) -> CandidateSets:
    want = set(candidate_set_ids)
    set_ids: List[int] = []
    costs: List[int] = []
    elems: List[List[int]] = []

    best_local_for_elem = [-1] * n
    best_cost_for_elem = [10**30] * n

    for set_id, cost, es in _iter_sets(path, m):
        if set_id not in want:
            continue
        local_idx = len(set_ids)
        set_ids.append(set_id)
        costs.append(cost)
        elems.append(es)
        for e in es:
            if cost < best_cost_for_elem[e]:
                best_cost_for_elem[e] = cost
                best_local_for_elem[e] = local_idx

    # Если для какого-то элемента не осталось кандидатного множества — фильтрация слишком жёсткая.
    missing = [e for e in range(n) if best_local_for_elem[e] == -1]
    if missing:
        raise ValueError(
            f"Candidate filtering lost coverage for {len(missing)} elements; "
            f"consider increasing per-element limit. First missing: {missing[:10]}"
        )

    return CandidateSets(
        set_ids=set_ids,
        costs=costs,
        elems=elems,
        inv=None,
        best_local_for_elem=best_local_for_elem,
    )


def _greedy_cover(
    n: int,
    cand: CandidateSets,
    elem_weight: Sequence[float],
    rng: random.Random,
    jitter: float,
    time_deadline: float,
) -> List[int]:
    uncovered = [True] * n
    remaining = n
    chosen = [False] * len(cand.set_ids)
    solution_local: List[int] = []

    heap: List[Tuple[float, int, float]] = []
    for i, es in enumerate(cand.elems):
        benefit = 0.0
        for e in es:
            benefit += elem_weight[e]
        if benefit <= 0:
            continue
        noise = 1.0
        if jitter > 0:
            noise += rng.uniform(-jitter, jitter)
        score = (cand.costs[i] / benefit) * noise
        heapq.heappush(heap, (score, i, benefit))

    while remaining > 0:
        if time.monotonic() >= time_deadline:
            break
        if not heap:
            break

        score, i, _old_benefit = heapq.heappop(heap)
        if chosen[i]:
            continue

        # Ленивый пересчёт: по мере покрытия элементов "выгода" множества уменьшается,
        # поэтому score в куче может устареть. Пересчитываем только при извлечении.
        benefit = 0.0
        for e in cand.elems[i]:
            if uncovered[e]:
                benefit += elem_weight[e]

        if benefit <= 0:
            continue

        noise = 1.0
        if jitter > 0:
            noise += rng.uniform(-jitter, jitter)
        new_score = (cand.costs[i] / benefit) * noise
        if new_score > score + 1e-12:
            heapq.heappush(heap, (new_score, i, benefit))
            continue

        chosen[i] = True
        solution_local.append(i)
        for e in cand.elems[i]:
            if uncovered[e]:
                uncovered[e] = False
                remaining -= 1

    # Достраиваем до допустимого решения (важно для тайм-лимита).
    if remaining > 0:
        for e in range(n):
            if not uncovered[e]:
                continue
            i = cand.best_local_for_elem[e]
            if i < 0:
                continue
            if not chosen[i]:
                chosen[i] = True
                solution_local.append(i)
                for ee in cand.elems[i]:
                    if uncovered[ee]:
                        uncovered[ee] = False
                        remaining -= 1
            if remaining <= 0:
                break

    if remaining != 0:
        raise RuntimeError("Failed to construct a feasible cover (internal error)")

    return solution_local


def _prune_redundant(n: int, cand: CandidateSets, solution_local: List[int]) -> List[int]:
    cover_count = [0] * n
    for i in solution_local:
        for e in cand.elems[i]:
            cover_count[e] += 1

    # Убираем избыточные множества, начиная с самых дорогих:
    # если все элементы множества покрыты хотя бы 2 раза, его можно удалить.
    solution_local_sorted = sorted(solution_local, key=lambda i: cand.costs[i], reverse=True)
    keep = set(solution_local)
    for i in solution_local_sorted:
        if i not in keep:
            continue
        es = cand.elems[i]
        if all(cover_count[e] >= 2 for e in es):
            keep.remove(i)
            for e in es:
                cover_count[e] -= 1

    return sorted(keep)


def _build_inv_index(n: int, cand: CandidateSets) -> List[List[int]]:
    inv: List[List[int]] = [[] for _ in range(n)]
    for i, es in enumerate(cand.elems):
        for e in es:
            inv[e].append(i)
    return inv


def _try_replace_expensive_sets(
    n: int,
    cand: CandidateSets,
    solution_local: List[int],
    time_deadline: float,
) -> List[int]:
    if cand.inv is None:
        cand.inv = _build_inv_index(n, cand)

    chosen = [False] * len(cand.set_ids)
    for i in solution_local:
        chosen[i] = True

    cover_count = [0] * n
    for i in solution_local:
        for e in cand.elems[i]:
            cover_count[e] += 1

    # Пробуем улучшать решение с конца: сначала самые дорогие множества.
    for i in sorted(solution_local, key=lambda x: cand.costs[x], reverse=True):
        if time.monotonic() >= time_deadline:
            break
        if not chosen[i]:
            continue
        # Элементы, которые покрываются только этим множеством.
        uniq = [e for e in cand.elems[i] if cover_count[e] == 1]
        if not uniq:
            continue

        need = set(uniq)
        replacement: List[int] = []
        replacement_cost = 0
        budget = cand.costs[i] - 1

        # Жадно докрываем need невыбранными множествами.
        while need and replacement_cost <= budget and time.monotonic() < time_deadline:
            best_j = -1
            best_score = float("inf")
            best_cov: List[int] = []

            sampled = list(need)
            if len(sampled) > 20:
                sampled = sampled[:20]

            seen_sets = set()
            for e in sampled:
                for j in cand.inv[e]:
                    if chosen[j] or j == i or j in seen_sets:
                        continue
                    seen_sets.add(j)
                    cov = [ee for ee in cand.elems[j] if ee in need]
                    if not cov:
                        continue
                    score = cand.costs[j] / len(cov)
                    if score < best_score:
                        best_score = score
                        best_j = j
                        best_cov = cov

            if best_j < 0:
                break

            replacement.append(best_j)
            replacement_cost += cand.costs[best_j]
            for ee in best_cov:
                need.discard(ee)

        if need:
            continue
        if replacement_cost >= cand.costs[i]:
            continue

        # Apply replacement.
        chosen[i] = False
        for e in cand.elems[i]:
            cover_count[e] -= 1

        for j in replacement:
            if chosen[j]:
                continue
            chosen[j] = True
            solution_local.append(j)
            for e in cand.elems[j]:
                cover_count[e] += 1

    improved = [i for i, v in enumerate(chosen) if v]
    return _prune_redundant(n, cand, improved)


def _objective(cand: CandidateSets, solution_local: Sequence[int]) -> int:
    return int(sum(cand.costs[i] for i in solution_local))


def solve(path: str, time_limit_s: float) -> Solution:
    start = time.monotonic()
    time_deadline = start + time_limit_s

    n, m = _read_header(path)
    #1)Отбираем кандидатов и считаем частоты элементов за один проход.
    per_elem_limit = _candidate_limit(n, m)
    freq, candidate_ids = _build_candidates(path, n, m, per_elem_limit)
    #2)Загружаем в память только кандидатов (а не все m множеств).
    cand = _load_candidate_sets(path, n, m, candidate_ids)

    inv_freq = [1.0 / max(1, f) for f in freq]

    # Небольшая рандомизация (jitter) и несколько прогонов часто дают лучшее качество,
    # чем один детерминированный запуск жадности.
    rng = random.Random((n * 1_000_003) ^ m)
    runs: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (0.5, 0.01),
        (1.0, 0.02),
    ]

    best_sol: Optional[List[int]] = None
    best_obj = 10**30

    for beta, jitter in runs:
        if time.monotonic() >= time_deadline:
            break
        if beta == 0.0:
            weights = [1.0] * n
        else:
            weights = [w**beta for w in inv_freq]

        sol = _greedy_cover(n, cand, weights, rng, jitter, time_deadline)
        sol = _prune_redundant(n, cand, sol)

        # Локальный поиск включаем только если есть запас времени и кандидатов не слишком много.
        if time.monotonic() + 0.5 < time_deadline and len(cand.set_ids) <= 250_000:
            sol = _try_replace_expensive_sets(n, cand, sol, time_deadline)

        obj = _objective(cand, sol)
        if obj < best_obj:
            best_obj = obj
            best_sol = sol

    if best_sol is None:
        raise RuntimeError("No solution produced")

    return Solution(
        set_ids=[cand.set_ids[i] for i in best_sol],
        objective=_objective(cand, best_sol),
        m=m,
    )


def _write_solution(solution: Solution, output_format: OutputFormat) -> None:
    out = sys.stdout
    if output_format == "indices":
        # Компактный формат: количество выбранных множеств и их индексы.
        out.write(str(len(solution.set_ids)) + "\n")
        if solution.set_ids:
            out.write(" ".join(map(str, sorted(solution.set_ids))) + "\n")
        else:
            out.write("\n")
        return

    if output_format == "coursera":
        out.write(f"{solution.objective} 0\n")
        chosen = bytearray(solution.m)
        for i in solution.set_ids:
            if 0 <= i < solution.m:
                chosen[i] = 1

        chunk_size = 200_000
        for start in range(0, solution.m, chunk_size):
            end = min(solution.m, start + chunk_size)
            out.write(" ".join("1" if chosen[i] else "0" for i in range(start, end)))
            out.write(" " if end < solution.m else "\n")
        return

    raise ValueError(f"Unsupported output format: {output_format!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", nargs="?", default=None)
    parser.add_argument("--time-limit", type=float, default=295.0)
    parser.add_argument("--output-format", choices=["indices", "coursera"], default="indices")
    args = parser.parse_args()

    inst = _open_instance(args)
    try:
        solution = solve(inst.path, args.time_limit)
        _write_solution(solution, args.output_format)
        return 0
    finally:
        if inst.delete_after:
            try:
                os.unlink(inst.path)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

