#!/usr/bin/env python3
"""Stamp out candidate submission files with one panel-width schedule enrolled.

Keeps `submission.py` itself clean: no env-var reads or runtime switches land
in the ranked source. Each candidate is a full copy of the current submission
with `_SPLIT32_NB_SCHEDULE` populated for exactly one shape, so paired B200
runs compare one enrolled shape against the untouched baseline.

Usage:
    python experiments/032-panel-width-schedule/make_candidates.py --list
    python experiments/032-panel-width-schedule/make_candidates.py 640x512 A
    python experiments/032-panel-width-schedule/make_candidates.py --all A

Writes candidate-<shape>-<variant>.py next to this script.
"""

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SUBMISSION = ROOT / "submission.py"

ANCHOR = "    _SPLIT32_NB_SCHEDULE = {}"

# variant -> {(batch, n): schedule}
SCHEDULES = {
    "A": {  # tail taper: only the late panels narrow
        (256, 128): (64, 32, 32),
        (64, 256): (128, 64, 32, 32),
        (16, 512): (128, 128, 128, 64, 32, 32),
        (640, 512): (128, 128, 128, 64, 32, 32),
        (4, 1024): (128,) * 7 + (64, 32, 32),
        (60, 1024): (128,) * 7 + (64, 32, 32),
        (8, 2048): (128,) * 15 + (64, 32, 32),
    },
    "B": {  # wide head plus taper
        (256, 128): (64, 64),
        (64, 256): (64, 64, 64, 64),
        (16, 512): (256, 128, 64, 32, 32),
        (640, 512): (256, 128, 64, 32, 32),
        (4, 1024): (256, 256, 128, 128, 128, 64, 32, 32),
        (60, 1024): (256, 256, 256, 128, 64, 32, 32),
        (8, 2048): (256,) * 7 + (128, 64, 32, 32),
    },
    # W (added after variant A refuted the taper): wide UNIFORM panels, no
    # taper. Variant A showed every extra panel costs the ~16us serial
    # launch floor (S27/S29) while the tapered tail processes almost no data,
    # so the only direction that can win is *fewer, wider* panels -- halving
    # the launch count. NB=256 doubles _trailing_nb's [TILE x NB] tile, which
    # risks register spill; that is exactly what this probe measures.
    # 256x128 is omitted: n=128 cannot hold a 256-wide panel and it is already
    # a single panel (A regressed it hardest by splitting it).
    "W": {
        (64, 256): (256,),
        (16, 512): (256, 256),
        (640, 512): (256, 256),
        (4, 1024): (256, 256, 256, 256),
        (60, 1024): (256, 256, 256, 256),
        (8, 2048): (256,) * 8,
    },
    # X: 8x2048-only wide-schedule tuning. Variant W showed 8x2048 is the one
    # shape where halving launches (NB=256) beats the NB=256 _trailing_nb spill
    # (1.031x). This pushes the same lever further -- NB=512 quarters the panel
    # count (4 panels) but quadruples the [TILE x NB] tile vs 128; and a 384-ish
    # middle ground is not a power of two, so 512 is the next rung. Also probes
    # a mixed (256-wide bulk, 512 head) in case the very first panels -- with the
    # most trailing rows to amortize -- prefer to be widest.
    "X": {
        (8, 2048): (512, 512, 512, 512),
    },
    "X2": {
        (8, 2048): (512, 512, 256, 256, 256, 256),
    },
}


def validate(shape, sched):
    """Same gate as _validate_nb_schedules() in submission.py."""
    n = shape[1]
    if sum(sched) != n:
        raise SystemExit(f"{shape} {sched}: sums to {sum(sched)}, expected {n}")
    for nb in sched:
        if nb < 32 or (nb & (nb - 1)) != 0:
            raise SystemExit(
                f"{shape} {sched}: width {nb} is not a power of two >= 32 "
                "(tl.arange bound in _trailing_nb)"
            )


def render(table):
    lines = ["    _SPLIT32_NB_SCHEDULE = {"]
    for shape, sched in table.items():
        lines.append(f"        {shape}: {sched!r},")
    lines.append("    }")
    return "\n".join(lines)


def _read_source():
    source = SUBMISSION.read_text()
    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"expected exactly one {ANCHOR!r} in submission.py; "
            f"found {source.count(ANCHOR)}. Anchor drifted -- update this script."
        )
    return source


def build(shape, variant):
    sched = SCHEDULES[variant][shape]
    validate(shape, sched)
    out = _read_source().replace(ANCHOR, render({shape: sched}))
    dest = HERE / f"candidate-{shape[0]}x{shape[1]}-{variant}.py"
    dest.write_text(out)
    return dest, sched


def build_combined(variant):
    """One candidate file that enrolls every shape of `variant` at once, so a
    single paired probe sweeps the whole variant. Each schedule is validated
    exactly as _validate_nb_schedules() does at import."""
    table = SCHEDULES[variant]
    for shape, sched in table.items():
        validate(shape, sched)
    out = _read_source().replace(ANCHOR, render(table))
    dest = HERE / f"candidate-all-{variant}.py"
    dest.write_text(out)
    return dest, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", nargs="?", help="e.g. 640x512")
    ap.add_argument("variant", nargs="?", default="A", choices=sorted(SCHEDULES))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", metavar="VARIANT", choices=sorted(SCHEDULES))
    ap.add_argument(
        "--combined",
        metavar="VARIANT",
        choices=sorted(SCHEDULES),
        help="stamp ONE file enrolling every shape of VARIANT (candidate-all-VARIANT.py)",
    )
    args = ap.parse_args()

    if args.combined:
        dest, table = build_combined(args.combined)
        print(f"{dest.name}: {len(table)} shapes enrolled")
        for shape, sched in table.items():
            print(f"  {shape[0]}x{shape[1]:<5} {sched}")
        return

    if args.list:
        for variant, table in sorted(SCHEDULES.items()):
            print(f"variant {variant}:")
            for shape, sched in table.items():
                validate(shape, sched)
                print(f"  {shape[0]}x{shape[1]:<5} {sched}")
        return

    if args.all:
        for shape in SCHEDULES[args.all]:
            dest, sched = build(shape, args.all)
            print(f"{dest.name}: {sched}")
        return

    if not args.shape:
        ap.error("give a shape (e.g. 640x512), --all VARIANT, or --list")

    m = re.fullmatch(r"(\d+)x(\d+)", args.shape)
    if not m:
        ap.error(f"bad shape {args.shape!r}; expected BATCHxN e.g. 640x512")
    shape = (int(m.group(1)), int(m.group(2)))
    if shape not in SCHEDULES[args.variant]:
        ap.error(f"no variant {args.variant} schedule for {shape}")

    dest, sched = build(shape, args.variant)
    print(f"{dest}: {sched}")


if __name__ == "__main__":
    sys.exit(main())
