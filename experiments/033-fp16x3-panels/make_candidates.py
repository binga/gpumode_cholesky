#!/usr/bin/env python3
"""Stamp candidate submissions that switch split32 PANEL dots from tf32x3 to
the fp16x3 three-fp16-MMA emulation (lever L4).

Baseline is the L2-banked source (`baseline-l4.py` == root submission.py at the
time exp 033 began: 8x2048 NB=256 schedule, all panels tf32x3). Each candidate
flips `panel_prec` (the first element of the `_SPLIT32_SHAPES` tuple) to
"fp16x3" for the enrolled shapes only; trailing precision is untouched.

Usage:
    python .../make_candidates.py --combined        # all 7 shapes -> fp16x3
    python .../make_candidates.py 640x512 8x2048     # a chosen subset
"""

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASELINE = HERE / "baseline-l4.py"

# Every split32 shape currently ships panel_prec="tf32x3".
ALL_SHAPES = [
    (256, 128), (64, 256), (16, 512), (640, 512),
    (4, 1024), (60, 1024), (8, 2048),
]


def flip(source, shape, prec):
    b, n = shape
    old = f'({b}, {n}): ("tf32x3"'
    new = f'({b}, {n}): ("{prec}"'
    if source.count(old) != 1:
        raise SystemExit(
            f"expected exactly one {old!r} in baseline-l4.py; "
            f"found {source.count(old)}. Anchor drifted."
        )
    return source.replace(old, new)


def build(shapes, prec, tag):
    source = BASELINE.read_text()
    for shape in shapes:
        source = flip(source, shape, prec)
    dest = HERE / f"candidate-{prec}-{tag}.py"
    dest.write_text(source)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes", nargs="*", help="e.g. 640x512 8x2048")
    ap.add_argument("--combined", action="store_true", help="all 7 shapes")
    ap.add_argument(
        "--prec",
        default="fp16x3",
        choices=["fp16x3", "tf32"],
        help="panel precision to switch to (default fp16x3). tf32 is native "
        "1-pass -- no fp16 register blowup, tests whether panels even need 3 passes.",
    )
    args = ap.parse_args()

    if args.combined:
        dest = build(ALL_SHAPES, args.prec, "all")
        print(f"{dest.name}: all 7 panels -> {args.prec}")
        return
    if not args.shapes:
        ap.error("give shapes (e.g. 640x512 8x2048) or --combined")
    shapes = []
    for s in args.shapes:
        m = re.fullmatch(r"(\d+)x(\d+)", s)
        if not m:
            ap.error(f"bad shape {s!r}")
        shapes.append((int(m.group(1)), int(m.group(2))))
    tag = "_".join(f"{b}x{n}" for b, n in shapes)
    dest = build(shapes, args.prec, tag)
    print(f"{dest.name}: {shapes} -> {args.prec} panels")


if __name__ == "__main__":
    sys.exit(main())
