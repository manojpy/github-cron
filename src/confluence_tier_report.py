#!/usr/bin/env python3
"""Win rate / net EV by confluence tier, from your own outcome archive.

Use it to check the policy knob ALERT_UNPROVEN_MIN_CONFLUENCE_PCT (default 90)
against real results instead of a hunch:

    python3 confluence_tier_report.py --data-dir /path/to/data-repo --days 60
    python3 confluence_tier_report.py --data-dir ... --shadow     # shadow rows
    python3 confluence_tier_report.py --data-dir ... --bar 90 --json

Read-only. Rows come from archive_reader.load_archived_outcomes, the same
loader the Brain uses, so the numbers match its view of the data.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

TIERS: Tuple[Tuple[str, float, float], ...] = (
    ("95-100%", 95.0, 100.01),
    ("90-95%", 90.0, 95.0),
    ("85-90%", 85.0, 90.0),
    ("80-85%", 80.0, 85.0),
    ("<80%", 0.0, 80.0),
)
MIN_N_FOR_VERDICT = 30


def _wilson_lo(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / denom)


def _stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    wins = sum(1 for r in rows if r.get("win"))
    evs = [float(r["net_pnl_pct"]) for r in rows if r.get("net_pnl_pct") is not None]
    return {
        "n": n,
        "wr": (wins / n) if n else None,
        "wilson_lo": _wilson_lo(wins, n) if n else None,
        "net_ev": (sum(evs) / len(evs)) if evs else None,
        "n_ev": len(evs),
    }


def tier_report(rows: Sequence[Dict[str, Any]], bar: float = 90.0) -> Dict[str, Any]:
    """Pure: group rows by confluence tier (and tier x direction)."""
    out: Dict[str, Any] = {"n_total": len(rows), "bar": bar, "tiers": {}, "by_direction": {}}
    for name, lo, hi in TIERS:
        sel = [r for r in rows if r.get("conf_pct") is not None and lo <= float(r["conf_pct"]) < hi]
        out["tiers"][name] = _stats(sel)
        for d in ("buy", "sell"):
            out["by_direction"].setdefault(d, {})[name] = _stats(
                [r for r in sel if str(r.get("direction")) == d]
            )
    above = [r for r in rows if r.get("conf_pct") is not None and float(r["conf_pct"]) >= bar]
    out["at_or_above_bar"] = _stats(above)
    out["verdict"] = _verdict(out["at_or_above_bar"])
    return out


def _verdict(s: Dict[str, Any]) -> str:
    if s["n"] < MIN_N_FOR_VERDICT:
        return (f"NOT ENOUGH DATA: only {s['n']} resolved rows at/above the bar "
                f"(need {MIN_N_FOR_VERDICT}+). Keep unproven TAKE small or off.")
    ev = s["net_ev"]
    if ev is None:
        return "NO net_pnl_pct on these rows; cannot judge edge."
    if ev > 0 and s["wr"] >= 0.5:
        return (f"SUPPORTED: WR {s['wr']:.0%} (Wilson low {s['wilson_lo']:.0%}), "
                f"net EV {ev:+.2f}% over n={s['n']}.")
    return (f"NOT SUPPORTED: WR {s['wr']:.0%}, net EV {ev:+.2f}% over n={s['n']}. "
            f"Raise the bar or disable ENABLE_ALERT_UNPROVEN_TAKE.")


def _fmt(s: Dict[str, Any]) -> str:
    if not s["n"]:
        return "      -"
    wr = f"{s['wr']:.0%}"
    lo = f"{s['wilson_lo']:.0%}"
    ev = f"{s['net_ev']:+.2f}%" if s["net_ev"] is not None else "   n/a"
    return f"{s['n']:>5}  {wr:>4}  {lo:>4}  {ev:>7}"


def render(rep: Dict[str, Any]) -> str:
    lines = [f"Confluence tier report — {rep['n_total']} resolved rows", "",
             "tier        n     WR  W-lo   netEV"]
    for name, _, _ in TIERS:
        lines.append(f"{name:<9} {_fmt(rep['tiers'][name])}")
    for d in ("buy", "sell"):
        lines.append("")
        lines.append(f"{d.upper()} only")
        for name, _, _ in TIERS:
            lines.append(f"{name:<9} {_fmt(rep['by_direction'][d][name])}")
    lines += ["", f"At or above {rep['bar']:.0f}%: {rep['verdict']}"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder that contains outcomes/ (and shadow/)")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--shadow", action="store_true", help="read shadow outcomes instead of real ones")
    ap.add_argument("--bar", type=float, default=90.0, help="confluence %% bar to judge (default 90)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from archive_reader import load_archived_outcomes
    rows = load_archived_outcomes(args.data_dir, window_days=args.days, shadow=args.shadow)
    rep = tier_report(rows, bar=args.bar)
    print(json.dumps(rep, indent=2) if args.json else render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
