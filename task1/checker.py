#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple


WORKDIR = Path(__file__).resolve().parent


EVALUATED_TESTS: List[Path] = [
    WORKDIR / "data" / "sc_157_0",
    WORKDIR / "data" / "sc_330_0",
    WORKDIR / "data" / "sc_1000_11",
    WORKDIR / "data" / "sc_5000_1",
    WORKDIR / "data" / "sc_10000_5",
    WORKDIR / "data" / "sc_10000_2",
]


@dataclass(frozen=True)
class InstanceHeader:
    n: int
    m: int


def read_header(path: Path) -> InstanceHeader:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip().split()
    if len(first) != 2:
        raise ValueError(f"Bad header in {path}: {first!r}")
    n, m = map(int, first)
    return InstanceHeader(n=n, m=m)


def parse_solution(stdout: str, m: int) -> List[int]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty solver output")

    first_parts = lines[0].split()
    if len(first_parts) == 2 and len(lines) >= 2:
        try:
            _obj = int(float(first_parts[0]))
            _flag = int(first_parts[1])
            vec = list(map(int, lines[1].split()))
            if len(vec) == m and all(v in (0, 1) for v in vec):
                return [i for i, v in enumerate(vec) if v == 1]
        except ValueError:
            pass

    if len(first_parts) == 1:
        try:
            k = int(first_parts[0])
            rest: List[int] = []
            for ln in lines[1:]:
                rest.extend(map(int, ln.split()))
            if k == len(rest):
                return rest
        except ValueError:
            pass

    all_ints: List[int] = []
    for ln in lines:
        all_ints.extend(map(int, ln.split()))
    return all_ints


def validate_and_score(instance_path: Path, chosen_set_ids: Sequence[int]) -> Tuple[bool, int, str]:
    header = read_header(instance_path)
    chosen: Set[int] = set(chosen_set_ids)

    if any(i < 0 or i >= header.m for i in chosen):
        bad = next(i for i in chosen if i < 0 or i >= header.m)
        return False, 0, f"Set index out of range: {bad} (m={header.m})"

    covered = [False] * header.n
    remaining = header.n
    obj = 0

    with instance_path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        for set_id in range(header.m):
            line = f.readline()
            if not line:
                return False, 0, f"Unexpected EOF at set {set_id}/{header.m}"
            if set_id not in chosen:
                continue
            parts = list(map(int, line.split()))
            if len(parts) < 2:
                return False, 0, f"Bad set line for chosen set {set_id}: {line!r}"
            cost = parts[0]
            obj += cost
            for e in parts[1:]:
                if 0 <= e < header.n and not covered[e]:
                    covered[e] = True
                    remaining -= 1

    if remaining != 0:
        return False, obj, f"Not all elements covered, missing={remaining}"

    return True, obj, "ok"


def run_solver(instance_path: Path, timeout_s: float) -> str:
    cmd = [sys.executable, str(WORKDIR / "solve.py"), str(instance_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if res.returncode != 0:
        raise RuntimeError(
            f"Solver failed on {instance_path.name} with code {res.returncode}\n"
            f"stderr:\n{res.stderr}\n"
            f"stdout:\n{res.stdout}\n"
        )
    return res.stdout


def main() -> int:
    timeout_s = 300.0
    total_obj = 0

    for p in EVALUATED_TESTS:
        if not p.exists():
            print(f"[SKIP] {p} not found")
            continue

        try:
            header = read_header(p)
        except Exception as e:
            print(f"[SKIP] {p.name}: cannot read header ({e})")
            continue

        try:
            t0 = time.monotonic()
            stdout = run_solver(p, timeout_s=timeout_s)
            dt = time.monotonic() - t0

            chosen = parse_solution(stdout, header.m)
            ok, obj, msg = validate_and_score(p, chosen)
        except subprocess.TimeoutExpired:
            ok, obj, msg = False, 0, "timeout"
            dt = timeout_s
        except Exception as e:
            ok, obj, msg = False, 0, f"checker error: {e}"
            dt = 0.0

        status = "OK" if ok else "FAIL"
        print(f"[{status}] {p.name}: obj={obj}, sets={len(set(chosen)) if ok else 0}, time={dt:.2f}s ({msg})")
        if ok:
            total_obj += obj

    print(f"TOTAL_OBJ (sum over OK): {total_obj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

