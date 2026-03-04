#!/usr/bin/env python3
"""
Метод: ветвление и границы с верхней оценкой по релаксации (дробный рюкзак).
"""

import sys
from typing import List, Tuple


def read_input(lines: List[str]) -> Tuple[int, int, List[Tuple[int, int]]]:
    """Читает n, W и список (v_i, w_i)."""
    n, W = map(int, lines[0].split())
    items = []
    for i in range(1, 1 + n):
        v, w = map(int, lines[i].split())
        items.append((v, w))
    return n, W, items


def fractional_knapsack_value(items: List[Tuple[int, int]], cap: int) -> float:
    """
    Стоимость оптимального дробного рюкзака (релаксация).
    Предполагается, что items уже отсортированы по v/w по убыванию.
    """
    if cap <= 0:
        return 0.0
    val = 0.0
    for v, w in items:
        if cap <= 0:
            break
        take = min(w, cap)
        val += v * (take / w)
        cap -= take
    return val


def solve_knapsack(n: int, W: int, items: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    """
    Решает 0/1 рюкзак методом ветвления и границ.
    Возвращает (максимальная стоимость, список x_i: 0 или 1).
    """
    if n == 0:
        return 0, []

    indexed = [(i, v, w) for i, (v, w) in enumerate(items)]
    indexed.sort(key=lambda x: x[1] / x[2] if x[2] > 0 else float('inf'), reverse=True)
    order = [x[0] for x in indexed]  # исходные индексы в порядке обхода
    sorted_items = [(x[1], x[2]) for x in indexed]

    best_value = 0
    best_take: List[int] = []  #в порядке order

    cur_w, cur_v = 0, 0
    greedy_take = []
    for v, w in sorted_items:
        if cur_w + w <= W:
            cur_w += w
            cur_v += v
            greedy_take.append(1)
        else:
            greedy_take.append(0)
    if cur_v > best_value:
        best_value = cur_v
        best_take = list(greedy_take)

    stack: List[Tuple[int, int, int, List[int]]] = [(0, 0, 0, [])]

    while stack:
        i, cur_val, cur_w, path = stack.pop()

        if i == n:
            if cur_val > best_value:
                best_value = cur_val
                best_take = list(path)
            continue

        v, w = sorted_items[i]
        rem_cap = W - cur_w

        bound = cur_val + fractional_knapsack_value(sorted_items[i + 1:], rem_cap)
        if bound <= best_value:
            continue

        stack.append((i + 1, cur_val, cur_w, path + [0]))

        if cur_w + w <= W:
            stack.append((i + 1, cur_val + v, cur_w + w, path + [1]))

    take_original = [0] * n
    for pos, orig_idx in enumerate(order):
        take_original[orig_idx] = best_take[pos]

    return best_value, take_original


def main():
    lines = sys.stdin.read().strip().split('\n')
    if not lines:
        print(0)
        print()
        return
    n, W, items = read_input(lines)
    value, solution = solve_knapsack(n, W, items)
    print(value)
    print(' '.join(map(str, solution)))


if __name__ == '__main__':
    main()
