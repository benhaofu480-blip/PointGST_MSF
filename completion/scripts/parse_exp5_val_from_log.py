#!/usr/bin/env python3
"""Parse Exp1-ON (crop02_04 seed42) validation Overall lines into epoch table."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = [
    ROOT
    / "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models"
    / "exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42/20260521_113321.log",
    ROOT
    / "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models"
    / "exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42/20260521_143225.log",
]

VAL_START = re.compile(r"\[VALIDATION\] Start validating epoch (\d+)")
OVERALL = re.compile(
    r"Overall\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def main() -> None:
    pending_epoch: int | None = None
    rows: list[tuple[int, float, float, float, float]] = []

    for log_path in LOGS:
        if not log_path.is_file():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m0 = VAL_START.search(line)
            if m0:
                pending_epoch = int(m0.group(1))
                continue
            m1 = OVERALL.search(line)
            if m1 and pending_epoch is not None and "\t0.000" in line:
                f, cdl1, cdl2, emd = map(float, m1.groups())
                rows.append((pending_epoch, f, cdl1, cdl2, emd))
                pending_epoch = None

    if not rows:
        print("No validation rows found.", file=sys.stderr)
        sys.exit(1)

    print("epoch\tF-Score\tCDL1\tCDL2\tEMD")
    for ep, f, cdl1, cdl2, emd in rows:
        print(f"{ep}\t{f:.3f}\t{cdl1:.3f}\t{cdl2:.3f}\t{emd:.3f}")

    best = min(rows, key=lambda r: r[2])
    print(f"\n# val-best epoch={best[0]} CDL1={best[2]:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
