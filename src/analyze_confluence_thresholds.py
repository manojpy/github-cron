#!/usr/bin/env python3
"""
analyze_confluence_thresholds.py

Reads the `outcome_log_stream` Redis stream (populated by the patched
resolve_pending_outcomes()) and reports win-rate as a function of the
confluence score each alert fired at. Use this to pick a data-driven
CONFLUENCE_MIN_SCORE instead of guessing.

Usage:
    export REDIS_URL="redis://..."          # same value as your GH secret
    python3 analyze_confluence_thresholds.py [--target-winrate 0.55] [--min-sample 20]

Notes:
  - Only outcomes resolved AFTER the score-logging patch was deployed will
    appear here. Old alert_stats:* counters are untouched and unaffected.
  - "score" is the weighted confluence score at fire time; "total" is the
    max possible that run (varies slightly per pair if a vote abstains).
  - Give this at least a few hundred resolved outcomes before trusting the
    numbers — with ~30 pairs firing occasionally, plan on running the bot
    for a few weeks before this is statistically meaningful.
"""
import argparse
import os
import sys
from collections import defaultdict

try:
    import redis
except ImportError:
    sys.exit("Missing dependency: pip install redis --break-system-packages")

STREAM_KEY = "outcome_log_stream"


def fetch_all(r: "redis.Redis"):
    """Pull every entry from the stream (XRANGE, paginated)."""
    entries = []
    last_id = "-"
    while True:
        batch = r.xrange(STREAM_KEY, min=last_id, count=1000)
        if not batch:
            break
        # xrange is inclusive of last_id, so skip the one we already have
        # after the first page
        start = 1 if entries and batch[0][0] == last_id else 0
        for entry_id, fields in batch[start:]:
            entries.append(fields)
        if len(batch) < 1000:
            break
        last_id = batch[-1][0]
    return entries


def to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-winrate", type=float, default=0.55,
                     help="Win rate you want the cap to guarantee (default 0.55, matches MIN_WIN_RATE)")
    ap.add_argument("--min-sample", type=int, default=20,
                     help="Minimum resolved outcomes required in a bucket before trusting it")
    ap.add_argument("--bucket-size", type=float, default=1.0,
                     help="Score bucket width for the breakdown table (default 1.0)")
    ap.add_argument("--pair", type=str, default=None,
                     help="Filter to a single pair, e.g. BTCUSD (default: all pairs combined)")
    args = ap.parse_args()

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        sys.exit("Set REDIS_URL in your environment first.")

    r = redis.from_url(redis_url, decode_responses=True)
    raw = fetch_all(r)
    if not raw:
        sys.exit(f"No entries found in '{STREAM_KEY}' yet. Deploy the patched bot and "
                  f"wait for outcomes to resolve (OUTCOME_LOOKAHEAD_CANDLES * 15min after each alert fires).")

    rows = []
    for f in raw:
        if args.pair and f.get("pair") != args.pair:
            continue
        score = to_float(f.get("score"))
        total = to_float(f.get("total"))
        win = f.get("win")
        if score is None or total is None or win is None or total <= 0:
            continue
        rows.append({
            "pair": f.get("pair"),
            "alert_key": f.get("alert_key"),
            "direction": f.get("direction"),
            "score": score,
            "total": total,
            "pct": score / total,
            "win": int(win),
        })

    if not rows:
        sys.exit("No usable rows after filtering (check --pair spelling?).")

    n = len(rows)
    overall_wr = sum(row["win"] for row in rows) / n
    print(f"\nLoaded {n} resolved outcomes"
          + (f" for {args.pair}" if args.pair else " across all pairs")
          + f". Overall win rate: {overall_wr:.1%}\n")

    # --- Table 1: win rate within each score bucket (independent bins) ---
    buckets = defaultdict(lambda: {"wins": 0, "n": 0})
    bw = args.bucket_size
    for row in rows:
        b = int(row["score"] // bw) * bw
        buckets[b]["n"] += 1
        buckets[b]["wins"] += row["win"]

    print(f"{'Score bucket':<16}{'N':>6}{'Win rate':>12}")
    print("-" * 34)
    for b in sorted(buckets):
        d = buckets[b]
        wr = d["wins"] / d["n"] if d["n"] else 0
        flag = "" if d["n"] >= args.min_sample else "  (low sample)"
        print(f"{b:>5.1f}-{b+bw:<9.1f}{d['n']:>6}{wr:>11.1%}{flag}")

    # --- Table 2: win rate for score >= X (cumulative, this is what a cap actually does) ---
    print(f"\n{'Cap (score >=)':<18}{'N passing':>12}{'Win rate':>12}")
    print("-" * 42)
    candidate_caps = sorted(set(row["score"] for row in rows))
    best_cap = None
    best_cap_n = None
    best_cap_wr = None
    for cap in candidate_caps:
        subset = [row for row in rows if row["score"] >= cap]
        if not subset:
            continue
        wr = sum(row["win"] for row in subset) / len(subset)
        marker = ""
        if len(subset) >= args.min_sample and wr >= args.target_winrate and best_cap is None:
            best_cap = cap
            best_cap_n = len(subset)
            best_cap_wr = wr
            marker = "  <-- lowest cap meeting target with enough sample"
        print(f"{cap:>10.1f}{len(subset):>12}{wr:>11.1%}{marker}")

    print()
    if best_cap is not None:
        print(f"Suggested CONFLUENCE_MIN_SCORE: {best_cap:.1f} (from {best_cap_n} samples, {best_cap_wr:.1%} win rate)")
        print(f"  (lowest score threshold with >= {args.min_sample} samples and >= {args.target_winrate:.0%} win rate)")
        print(f"  Raising the cap further trades fewer/later alerts for a possibly higher win rate — "
              f"check the table above for that trade-off.")
    else:
        print(f"No cap yet reaches your target win rate with sufficient sample size ({n} total samples collected).")
        print("Either lower --target-winrate, lower --min-sample, or collect more data.")

if __name__ == "__main__":
    main()
