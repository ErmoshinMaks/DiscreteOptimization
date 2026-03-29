#!/usr/bin/env python3

import sys

import solver


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: verify_output.py <graph_file> <solver_output_file>", file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        data = f.read()
    with open(sys.argv[2], encoding="utf-8") as f:
        out = f.read().strip().splitlines()
    n, adj = solver.parse_input(data)
    if len(out) < 2:
        print("bad output: need at least 2 lines", file=sys.stderr)
        sys.exit(1)
    claimed = int(out[0].split()[0])
    colors = list(map(int, out[1].split()))
    if len(colors) != n:
        print(f"bad size: expected {n} colors", file=sys.stderr)
        sys.exit(1)
    if not solver.is_proper(colors, adj):
        print("invalid: edge monochromatic", file=sys.stderr)
        sys.exit(1)
    actual = len(set(colors))
    if actual != claimed:
        print(f"warning: claimed {claimed} distinct colors, actual {actual}", file=sys.stderr)
    print("ok", "distinct", actual)


if __name__ == "__main__":
    main()
