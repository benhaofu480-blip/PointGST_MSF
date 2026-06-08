#!/usr/bin/env python3
"""Parse training log: val metrics, early-stop counter, ckpt-best epoch."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

VAL_RE = re.compile(
    r"\[Validation\] EPOCH:\s*(\d+)\s+Metrics = \['([\d.]+)', '([\d.]+)', '([\d.]+)'"
)
EARLY_RE = re.compile(r"\[Early Stop\] No improvement for (\d+) epochs, stopping at epoch (\d+)")
BEST_CKPT_RE = re.compile(r"Save checkpoint at .*/(ckpt-best\.pth)")
VAL_LINE_RE = re.compile(r"\[VALIDATION\] Start validating epoch (\d+)")


def parse_log(path: Path):
    rows = []
    early = None
    best_epochs = []
    last_val_epoch = None

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                ep = int(m.group(1))
                f_score, cdl1, cdl2 = map(float, m.groups()[1:])
                rows.append({"epoch": ep, "F-Score": f_score, "CDL1": cdl1, "CDL2": cdl2})
            m = EARLY_RE.search(line)
            if m:
                early = {"no_improve_epochs": int(m.group(1)), "stop_epoch": int(m.group(2))}
            if BEST_CKPT_RE.search(line):
                # previous validation epoch is the one that triggered save
                if rows:
                    best_epochs.append(rows[-1]["epoch"])
            m = VAL_LINE_RE.search(line)
            if m:
                last_val_epoch = int(m.group(1))

    return rows, early, best_epochs


def simulate_early_stop(rows, metric: str = "CDL1", val_freq: int = 10, patience: int = 30):
    """Reproduce runner.py logic (lower CDL1 is better)."""
    if not rows:
        return None
    lower_better = metric != "F-Score"
    best_val = None
    best_ep = None
    no_improve = 0
    history = []

    for r in rows:
        ep = r["epoch"]
        v = r[metric]
        if best_val is None:
            improved = True
        else:
            improved = (v < best_val) if lower_better else (v > best_val)
        if improved:
            best_val = v
            best_ep = ep
            no_improve = 0
            action = "NEW_BEST -> reset counter"
        else:
            no_improve += val_freq
            action = f"no improve (+{val_freq}) -> counter={no_improve}"
        history.append((ep, v, no_improve, action))
        if no_improve >= patience:
            return {
                "stop_epoch": ep,
                "best_epoch": best_ep,
                "best_value": best_val,
                "history": history,
            }
    return {
        "stop_epoch": None,
        "best_epoch": best_ep,
        "best_value": best_val,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "log",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "logs" / "msf_hard_ft_cov005_hardval_seed42.log",
    )
    ap.add_argument("--metric", default="CDL1", choices=["CDL1", "F-Score", "CDL2"])
    ap.add_argument("--val-freq", type=int, default=10)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"Log not found: {args.log}")
        return 1

    rows, early, best_epochs = parse_log(args.log)
    print(f"Log: {args.log}")
    print(f"Validation points: {len(rows)}")
    print()

    if not rows:
        print("No [Validation] EPOCH lines found.")
        return 1

    print(f"{'epoch':>6}  {'F-Score':>8}  {'CDL1':>8}  {'CDL2':>8}  note")
    print("-" * 50)
    best_cdl1 = min(r["CDL1"] for r in rows)
    best_f = max(r["F-Score"] for r in rows)
    for r in rows:
        notes = []
        if abs(r["CDL1"] - best_cdl1) < 1e-4:
            notes.append("best CDL1")
        if abs(r["F-Score"] - best_f) < 1e-4:
            notes.append("best F")
        if r["epoch"] in best_epochs:
            notes.append("ckpt-best saved")
        print(f"{r['epoch']:>6}  {r['F-Score']:>8.4f}  {r['CDL1']:>8.4f}  {r['CDL2']:>8.4f}  {', '.join(notes)}")

    print()
    sim = simulate_early_stop(rows, args.metric, args.val_freq, args.patience)
    print(f"Early-stop rule: metric={args.metric}, val_freq={args.val_freq}, patience={args.patience}")
    print(f"  (each failed val adds +{args.val_freq} to counter; stop when counter >= {args.patience})")
    print()
    for ep, v, cnt, action in sim["history"]:
        print(f"  epoch {ep:2d}: {args.metric}={v:.4f}  |  {action}")
    print()
    print(f"Simulated best: epoch {sim['best_epoch']}, {args.metric}={sim['best_value']:.4f}")
    if sim["stop_epoch"] is not None:
        print(f"Simulated stop: epoch {sim['stop_epoch']}")
    if early:
        print(f"Log early stop:   epoch {early['stop_epoch']}, counter={early['no_improve_epochs']}")

    # F vs CDL1 mismatch
    sim_f = simulate_early_stop(rows, "F-Score", args.val_freq, args.patience)
    if sim_f["best_epoch"] != sim["best_epoch"]:
        print()
        print("Note: F-Score best epoch differs from CDL1 best epoch:")
        print(f"  CDL1 best @ epoch {sim['best_epoch']}")
        print(f"  F-Score best @ epoch {sim_f['best_epoch']} (value {sim_f['best_value']:.4f})")
        if sim_f["stop_epoch"] is None:
            print("  If consider_metric were F-Score, training would NOT have early-stopped yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
