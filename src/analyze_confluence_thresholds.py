#!/usr/bin/env python3
"""
analyze_confluence_thresholds.py  v2.2 — Intelligent Threshold Advisor

Reads the outcome_log_stream Redis stream and produces a data-driven,
statistically-rigorous CONFLUENCE_MIN_ABS_SCORE recommendation.

Critical fixes in v2.2:
  • pct_move is now normalised to absolute favourable-move magnitude,
    fixing the buy/sell sign-cancellation bug that destroyed EV and R:R.
  • detect_temporal_drift uses wall-clock time (time.time()), not the
    last trade timestamp, so offline periods don't stale the window.
  • rr_ratio guards against near-zero loss magnitude.
  • smooth() / knee-point bails out on too-few data points.

Improvements over v1:
  1. Toxic-zone detection  — identifies score buckets with WR < 50%
  2. Knee-point detection  — finds where marginal WR gain per +1 score drops
     (smoothed with a 3-point rolling average to resist single-bucket noise)
  3. Wilson confidence intervals — statistical significance, not raw ratios
  4. Expected-value + R:R analysis — accounts for avg win/loss magnitude
  5. Temporal drift check  — is the edge decaying over recent weeks?
  6. Per-alert-key breakdown — which alert types are dragging scores down?
  7. Anomalous-bucket flagging — a dip surrounded by two stronger buckets
     gets called out for investigation instead of silently trusted
  8. Vote-combination breakdown (ported from v1's --vote-breakdown-range)
  9. Sample-size sanity banner — warns when the whole dataset, or the
     recommended threshold's own subset, is thin
  10. Multi-tier recommendation with clear reasoning

v2.1 fix: toxic-zone detection is informational only — it no longer forces
a hard floor on the final recommendation.

v2.2 additions:
  • --direction filter (buy / sell) with side-by-side summary
  • --alert-key filter
  • Per-pair breakdown
  • Global vote-importance ranking (which votes add edge vs noise)
  • Alert-frequency impact (alerts/week before vs after threshold)
  • Recommendation stability guidance
  • --json output mode for CI/CD automation
  • Guard-rails: no 0.0 recommendation, negative-EV warning,
    smooth/knee bail-out on small N

Usage:
    export REDIS_URL="redis://..."
    python3 analyze_confluence_thresholds.py [--target-winrate 0.55] [--min-sample 20]
                                              [--vote-breakdown-range 24,25]
                                              [--direction buy] [--json]
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

try:
    import redis
except ImportError:
    sys.exit("Missing dependency: pip install redis")

STREAM_KEY = "outcome_log_stream"


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def fetch_all(r: "redis.Redis"):
    entries = []
    last_id = "-"
    while True:
        batch = r.xrange(STREAM_KEY, min=last_id, count=1000)
        if not batch:
            break
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


def load_rows(r, pair_filter=None, direction_filter=None, alert_key_filter=None):
    raw = fetch_all(r)
    if not raw:
        print(f"No entries in '{STREAM_KEY}' yet.", file=sys.stderr)
        sys.exit(2)
    rows = []
    for f in raw:
        if pair_filter and f.get("pair") != pair_filter:
            continue
        if direction_filter and f.get("direction") != direction_filter:
            continue
        if alert_key_filter and f.get("alert_key") != alert_key_filter:
            continue
        score = to_float(f.get("score"))
        total = to_float(f.get("total"))
        win = f.get("win")
        if score is None or total is None or win is None or total <= 0:
            continue
        votes_raw = f.get("votes")
        try:
            votes = json.loads(votes_raw) if votes_raw else None
        except (TypeError, ValueError):
            votes = None
        rows.append({
            "pair": f.get("pair"),
            "alert_key": f.get("alert_key"),
            "direction": f.get("direction"),
            "score": score,
            "total": total,
            "pct": score / total,
            "win": int(win),
            "pct_move": to_float(f.get("pct_move"), 0.0),
            "entry_ts": int(f.get("entry_ts", 0)),
            "votes": votes,
        })
    if not rows:
        print("No usable rows after filtering.", file=sys.stderr)
        sys.exit(2)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────

def favourable_move(row):
    """Convert signed pct_move to absolute favourable-move magnitude.

    pct_move is signed by price direction, not by trade outcome:
        Buy win  -> pct_move = +0.5%
        Sell win -> pct_move = -0.5%
    Averaging signed wins across directions cancels out to ~0.
    We always want the magnitude of the move that was favourable
    to the position, so we take abs().
    """
    return abs(row["pct_move"])


# ──────────────────────────────────────────────────────────────────────
# Statistical helpers
# ──────────────────────────────────────────────────────────────────────

def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Wilson score interval — reliable even for small n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, centre - margin), min(1, centre + margin), p


def expected_value(wins, losses, avg_win_pct, avg_loss_pct):
    """EV per trade in % terms. Positive = profitable long-run."""
    n = wins + losses
    if n == 0:
        return 0.0
    wr = wins / n
    return wr * avg_win_pct - (1 - wr) * abs(avg_loss_pct)


def rr_ratio(avg_win_pct, avg_loss_pct):
    """Reward:risk ratio. None if there's no loss magnitude to divide by."""
    if not avg_loss_pct or abs(avg_loss_pct) < 1e-6:
        return None
    return avg_win_pct / abs(avg_loss_pct)


def format_rr(rr):
    return f"{rr:.2f}" if rr is not None else "n/a"


def smooth(values, window=3):
    """Simple centered moving average, used only to de-noise knee-point
    detection — never used for the numbers actually printed to the user."""
    n = len(values)
    if n < window:
        return values[:]
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


# ──────────────────────────────────────────────────────────────────────
# Analysis engines
# ──────────────────────────────────────────────────────────────────────

def detect_toxic_zones(rows, bucket_size=1.0, min_sample=10):
    """Find score buckets where win rate is consistently below 50%.
    Informational only — see module docstring on why this is no longer
    used as a hard floor on the recommendation."""
    buckets = defaultdict(lambda: {"wins": 0, "n": 0})
    for row in rows:
        b = int(row["score"] // bucket_size) * bucket_size
        buckets[b]["n"] += 1
        buckets[b]["wins"] += row["win"]

    toxic = []
    for b in sorted(buckets):
        d = buckets[b]
        if d["n"] < min_sample:
            continue
        wr = d["wins"] / d["n"]
        lo, hi, _ = wilson_ci(d["wins"], d["n"])
        if hi < 0.50:  # even the UPPER confidence bound is below 50%
            toxic.append((b, b + bucket_size, wr, d["n"]))
    return toxic


def detect_anomalous_buckets(buckets, bucket_size, min_sample=10, drop_pct=0.15):
    """Flag a bucket whose WR sits well below BOTH of its immediate
    neighbors — a dip surrounded by strength, worth investigating rather
    than silently trusting (e.g. a single bad alert-type polluting one
    score band)."""
    sorted_b = sorted(buckets.keys())
    anomalies = []
    for idx, b in enumerate(sorted_b):
        if idx == 0 or idx == len(sorted_b) - 1:
            continue
        prev_b, next_b = sorted_b[idx - 1], sorted_b[idx + 1]
        # only compare truly adjacent buckets (no gaps from empty bins)
        if abs(prev_b + bucket_size - b) > 1e-9 or abs(b + bucket_size - next_b) > 1e-9:
            continue
        d, dp, dn = buckets[b], buckets[prev_b], buckets[next_b]
        if d["n"] < min_sample or dp["n"] < min_sample or dn["n"] < min_sample:
            continue
        wr, wr_prev, wr_next = d["wins"] / d["n"], dp["wins"] / dp["n"], dn["wins"] / dn["n"]
        if wr_prev - wr >= drop_pct and wr_next - wr >= drop_pct:
            anomalies.append((b, b + bucket_size, wr, wr_prev, wr_next, d["n"]))
    return anomalies


def find_knee_point(caps_data, min_sample=30, smooth_window=3):
    """Find the knee where marginal WR gain per +1 score flattens.
    WR values are smoothed first to resist single-bucket noise (e.g. a
    small dip at one score level shouldn't manufacture a false knee).
    Returns the (unsmoothed) score at the knee."""
    if len(caps_data) < max(6, smooth_window * 2):
        return None
    wrs = [c[2] for c in caps_data]
    smoothed_wrs = smooth(wrs, window=smooth_window)

    best_knee = None
    best_ratio = 0.0
    for i in range(1, len(caps_data) - 1):
        prev_cap, prev_n, _ = caps_data[i - 1]
        curr_cap, curr_n, _ = caps_data[i]
        next_cap, next_n, _ = caps_data[i + 1]
        prev_wr, curr_wr, next_wr = smoothed_wrs[i - 1], smoothed_wrs[i], smoothed_wrs[i + 1]
        marginal_before = (curr_wr - prev_wr) / max(curr_cap - prev_cap, 0.01)
        marginal_after = (next_wr - curr_wr) / max(next_cap - curr_cap, 0.01)
        # Knee = where marginal gain drops most sharply
        drop = marginal_before - marginal_after
        if drop > best_ratio and curr_n >= min_sample:
            best_ratio = drop
            best_knee = curr_cap
    return best_knee


def detect_temporal_drift(rows, window_days=14):
    """Compare win rate of recent outcomes vs older ones.
    Uses wall-clock time (time.time()) as "now", NOT max(entry_ts),
    so bot downtime doesn't stale the window."""
    if not rows:
        return None, None, None
    now_ts = int(time.time())
    cutoff = now_ts - (window_days * 86400)
    recent = [r for r in rows if r["entry_ts"] >= cutoff]
    older = [r for r in rows if r["entry_ts"] < cutoff]
    if len(recent) < 10 or len(older) < 10:
        return None, None, None
    recent_wr = sum(r["win"] for r in recent) / len(recent)
    older_wr = sum(r["win"] for r in older) / len(older)
    return recent_wr, older_wr, len(recent)


def per_alert_breakdown(rows, min_sample=10):
    """Win rate by alert_key, pooled across pairs."""
    stats = defaultdict(lambda: {"wins": 0, "n": 0, "scores": []})
    for r in rows:
        s = stats[r["alert_key"]]
        s["wins"] += r["win"]
        s["n"] += 1
        s["scores"].append(r["score"])
    results = []
    for ak, s in stats.items():
        if s["n"] < min_sample:
            continue
        wr = s["wins"] / s["n"]
        avg_score = sum(s["scores"]) / len(s["scores"])
        results.append((ak, wr, s["n"], avg_score))
    results.sort(key=lambda x: x[1])
    return results


def per_pair_breakdown(rows, min_sample=10):
    """Win rate by trading pair."""
    stats = defaultdict(lambda: {"wins": 0, "n": 0})
    for r in rows:
        s = stats[r["pair"]]
        s["wins"] += r["win"]
        s["n"] += 1
    results = []
    for pair, s in stats.items():
        if s["n"] < min_sample:
            continue
        results.append((pair, s["wins"] / s["n"], s["n"]))
    results.sort(key=lambda x: x[1])
    return results


def vote_importance(rows, min_sample=10):
    """For each vote, win rate when it's True vs when it's False.
    Sorted by lift (wr_with - wr_without), descending."""
    vote_names = set()
    for r in rows:
        if r["votes"]:
            vote_names.update(r["votes"].keys())
    results = []
    for vn in sorted(vote_names):
        with_vote = [r for r in rows if r["votes"] and r["votes"].get(vn) is True]
        without_vote = [r for r in rows if r["votes"] and r["votes"].get(vn) is False]
        if len(with_vote) < min_sample or len(without_vote) < min_sample:
            continue
        wr_with = sum(r["win"] for r in with_vote) / len(with_vote)
        wr_without = sum(r["win"] for r in without_vote) / len(without_vote)
        results.append((vn, wr_with, len(with_vote), wr_without, len(without_vote), wr_with - wr_without))
    results.sort(key=lambda x: -x[5])
    return results


def compute_ev_by_cap(rows, caps, min_sample=10):
    """Expected value (and R:R) per trade for each cap level.
    Uses abs(pct_move) so buy and sell wins don't cancel."""
    ev_data = []
    for cap in caps:
        subset = [r for r in rows if r["score"] >= cap]
        if len(subset) < min_sample:
            continue
        wins = [favourable_move(r) for r in subset if r["win"]]
        losses = [favourable_move(r) for r in subset if not r["win"]]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        wr = len(wins) / len(subset)
        ev = expected_value(len(wins), len(losses), avg_win, avg_loss)
        ev_data.append((cap, len(subset), wr, ev, avg_win, avg_loss))
    return ev_data


def vote_combo_breakdown(rows, lo, hi, min_sample=20):
    """Ported from v1 --vote-breakdown-range: within a score band, win
    rate by which votes actually fired."""
    band_rows = [row for row in rows if lo <= row["score"] < hi and row["votes"] is not None]
    if not band_rows:
        return None
    combo_stats = defaultdict(lambda: {"wins": 0, "n": 0})
    for row in band_rows:
        combo = tuple(sorted(k for k, v in row["votes"].items() if v))
        combo_stats[combo]["n"] += 1
        combo_stats[combo]["wins"] += row["win"]
    return band_rows, combo_stats


def direction_split_summary(rows, min_sample=10):
    """Return (buy_wr, buy_n, sell_wr, sell_n) for side-by-side display."""
    buys = [r for r in rows if r["direction"] == "buy"]
    sells = [r for r in rows if r["direction"] == "sell"]
    buy_wr = sum(r["win"] for r in buys) / len(buys) if buys else None
    sell_wr = sum(r["win"] for r in sells) / len(sells) if sells else None
    return buy_wr, len(buys), sell_wr, len(sells)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Intelligent Confluence Threshold Advisor v2.2")
    ap.add_argument("--target-winrate", type=float, default=0.55)
    ap.add_argument("--min-sample", type=int, default=20)
    ap.add_argument("--bucket-size", type=float, default=1.0)
    ap.add_argument("--pair", type=str, default=None)
    ap.add_argument("--direction", type=str, default=None, choices=["buy", "sell"],
                    help="Filter to one direction only")
    ap.add_argument("--alert-key", type=str, default=None,
                    help="Filter to a single alert_key, e.g. 'ppo_signal_up'")
    ap.add_argument("--vote-breakdown-range", type=str, default=None,
                    help="Score range 'LOW,HIGH' to break down by vote combination, e.g. '24,25'.")
    ap.add_argument("--json", action="store_true",
                    help="Output recommendation as JSON to stdout")
    args = ap.parse_args()

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        sys.exit("Set REDIS_URL in your environment first.")

    r = redis.from_url(redis_url, decode_responses=True)
    rows = load_rows(r, args.pair, args.direction, args.alert_key)
    n = len(rows)
    overall_wr = sum(row["win"] for row in rows) / n

    # Direction split (always computed, even when filtered)
    buy_wr, buy_n, sell_wr, sell_n = direction_split_summary(rows)

    print(f"\n{'='*70}")
    print(f"  INTELLIGENT CONFLUENCE THRESHOLD ANALYSIS")
    print(f"{'='*70}")
    print(f"  Samples: {n} | Overall WR: {overall_wr:.1%} | "
          f"Target: {args.target_winrate:.0%} | Min sample: {args.min_sample}")
    if buy_wr is not None and sell_wr is not None:
        print(f"  Buy WR: {buy_wr:.1%} ({buy_n}) | Sell WR: {sell_wr:.1%} ({sell_n})")
    if n < 500:
        print(f"  ⚠️  Sample size is thin ({n} < 500) — treat every recommendation")
        print(f"      below as provisional until more outcomes resolve.")
    print(f"{'='*70}\n")

    # ── 1. Score bucket breakdown ──
    bw = args.bucket_size
    buckets = defaultdict(lambda: {"wins": 0, "n": 0})
    for row in rows:
        b = int(row["score"] // bw) * bw
        buckets[b]["n"] += 1
        buckets[b]["wins"] += row["win"]

    avg_per_bucket = n / len(buckets) if buckets else 0
    print(f"{'Score bucket':<16}{'N':>6}{'Win rate':>12}{'95% CI':>18}{'Verdict':>12}")
    print("-" * 64)
    for b in sorted(buckets):
        d = buckets[b]
        wr = d["wins"] / d["n"] if d["n"] else 0
        lo, hi, _ = wilson_ci(d["wins"], d["n"])
        if d["n"] < args.min_sample:
            verdict = "low sample"
        elif hi < 0.50:
            verdict = "🔴 TOXIC"
        elif lo >= 0.60:
            verdict = "🟢 STRONG"
        elif wr >= args.target_winrate:
            verdict = "🟡 viable"
        else:
            verdict = "🟠 weak"
        print(f"  {b:>5.1f}-{b+bw:<9.1f}{d['n']:>6}{wr:>11.1%}"
              f"  [{lo:.0%}-{hi:.0%}]{verdict:>12}")

    if avg_per_bucket < 15:
        wider = bw * 2
        print(f"\n💡 Average {avg_per_bucket:.0f} samples/bucket at width {bw:.1f} — buckets are noisy.")
        print(f"   Try --bucket-size {wider:.1f} to reduce per-bucket variance.")

    # ── 2. Toxic zone detection (informational — see v2.1 note) ──
    toxic_zones = detect_toxic_zones(rows, bw, min_sample=args.min_sample)
    if toxic_zones:
        print(f"\n⚠️  TOXIC ZONES DETECTED (WR statistically below 50%):")
        for lo_b, hi_b, wr, cnt in toxic_zones:
            print(f"   Score {lo_b:.1f}-{hi_b:.1f}: {wr:.0%} WR over {cnt} samples")
        print(f"   These are flagged for awareness. They do NOT automatically raise")
        print(f"   the recommended threshold below — a toxic bucket sandwiched between")
        print(f"   two good ones just means that alert combination is worth investigating,")
        print(f"   not that every score below it is unsafe (see cumulative table below).")
    else:
        print(f"\n✅ No statistically toxic zones detected.")

    # ── 3. Anomalous buckets (dip surrounded by strength) ──
    anomalies = detect_anomalous_buckets(buckets, bw, min_sample=args.min_sample)
    if anomalies:
        print(f"\n🔎 ANOMALOUS BUCKETS (dip surrounded by stronger neighbors — investigate):")
        for lo_b, hi_b, wr, wr_prev, wr_next, cnt in anomalies:
            print(f"   Score {lo_b:.1f}-{hi_b:.1f}: {wr:.1%} WR (N={cnt}), "
                  f"vs {wr_prev:.1%} below and {wr_next:.1%} above.")
            print(f"      → Check the per-alert breakdown below for what's firing in this band,")
            print(f"        or run --vote-breakdown-range {lo_b:.0f},{hi_b:.0f} to see which vote")
            print(f"        combination is dragging this bucket down.")

    # ── 4. Cumulative cap table with EV + R:R ──
    candidate_caps = sorted(set(row["score"] for row in rows))
    caps_data = []
    print(f"\n{'Cap':>8}{'N':>8}{'WR':>10}{'95% CI':>18}{'EV/trade':>12}{'R:R':>8}")
    print("-" * 64)
    for cap in candidate_caps:
        subset = [row for row in rows if row["score"] >= cap]
        if not subset or len(subset) < 5:
            continue
        wins_count = sum(r["win"] for r in subset)
        wr = wins_count / len(subset)
        lo, hi, _ = wilson_ci(wins_count, len(subset))
        wins_pct = [favourable_move(r) for r in subset if r["win"]]
        loss_pct = [favourable_move(r) for r in subset if not r["win"]]
        avg_w = sum(wins_pct) / len(wins_pct) if wins_pct else 0
        avg_l = sum(loss_pct) / len(loss_pct) if loss_pct else 0
        ev = expected_value(wins_count, len(subset) - wins_count, avg_w, avg_l)
        rr = rr_ratio(avg_w, avg_l)
        caps_data.append((cap, len(subset), wr, lo))
        print(f"  {cap:>6.1f}{len(subset):>8}{wr:>9.1%}"
              f"  [{lo:.0%}-{hi:.0%}]{ev:>+10.3f}%{format_rr(rr):>8}")

    # caps_data_simple keeps the (cap, n, wr) shape the knee function expects
    caps_data_simple = [(c[0], c[1], c[2]) for c in caps_data]

    # Guard: no caps_data means we can't recommend anything
    if not caps_data:
        print("\n❌ No cap levels have enough data. Cannot recommend a threshold.")
        sys.exit(2)

    # ── 5. Knee point (smoothed) ──
    knee = find_knee_point(caps_data_simple, min_sample=args.min_sample)
    if knee is not None:
        knee_row = next((c for c in caps_data if c[0] == knee), None)
        if knee_row:
            print(f"\n📐 Knee point detected at score {knee:.1f} "
                  f"(WR={knee_row[2]:.1%}, N={knee_row[1]})")
            print(f"   Beyond this point, each +1 score adds diminishing WR improvement.")

    # ── 6. EV-optimal threshold ──
    ev_data = compute_ev_by_cap(rows, candidate_caps, min_sample=args.min_sample)
    best_ev = max(ev_data, key=lambda x: x[3]) if ev_data else None
    if best_ev:
        rr = rr_ratio(best_ev[4], best_ev[5])
        print(f"\n💰 EV-optimal threshold: score >= {best_ev[0]:.1f}")
        print(f"   WR={best_ev[2]:.1%} | N={best_ev[1]} | "
              f"EV={best_ev[3]:+.3f}% per trade | R:R={format_rr(rr)} "
              f"(avg win={best_ev[4]:+.2f}%, avg loss={best_ev[5]:+.2f}%)")
        if best_ev[3] <= 0:
            print(f"   🚨 WARNING: EV-optimal threshold has non-positive EV. "
                  f"The strategy may be unprofitable across all score levels.")

    # ── 7. Temporal drift ──
    recent_wr, older_wr, recent_n = detect_temporal_drift(rows)
    if recent_wr is not None:
        drift = recent_wr - older_wr
        drift_icon = "📈" if drift > 0.02 else ("📉" if drift < -0.02 else "➡️")
        print(f"\n{drift_icon} Temporal drift: last 14d WR={recent_wr:.1%} ({recent_n} samples) "
              f"vs prior WR={older_wr:.1%} (Δ{drift:+.1%})")
        if drift < -0.05:
            print(f"   ⚠️  Edge may be decaying — consider raising threshold as a hedge.")

    # ── 8. Per-pair breakdown ──
    pair_stats = per_pair_breakdown(rows, min_sample=args.min_sample)
    if pair_stats:
        print(f"\n{'Pair':<12}{'WR':>8}{'N':>6}")
        print("-" * 28)
        for pair, wr, cnt in pair_stats:
            flag = " 🔴" if wr < 0.45 else (" 🟢" if wr >= 0.70 else "")
            print(f"  {pair:<10}{wr:>7.1%}{cnt:>6}{flag}")

    # ── 9. Per-alert breakdown ──
    alert_stats = per_alert_breakdown(rows, min_sample=args.min_sample)
    if alert_stats:
        print(f"\n{'Alert key':<30}{'WR':>8}{'N':>6}{'Avg score':>12}")
        print("-" * 56)
        # Show worst 5 and best 5

        if len(alert_stats) <= 10:
            display = alert_stats
        else:
            display = alert_stats[:5] + alert_stats[-5:]
        for idx, (ak, wr, cnt, avg_s) in enumerate(display):
            if len(alert_stats) > 10 and idx == 5:
                print(f"  ... ({len(alert_stats) - 10} more) ...")
            flag = " 🔴" if wr < 0.45 else (" 🟢" if wr >= 0.70 else "")
            print(f"  {ak:<28}{wr:>7.1%}{cnt:>6}{avg_s:>11.1f}{flag}")

    # ── 10. Vote importance ranking ──
    vote_imp = vote_importance(rows, min_sample=args.min_sample)
    if vote_imp:
        print(f"\n{'Vote':<28}{'WR (True)':>10}{'N':>6}{'WR (False)':>12}{'N':>6}{'Lift':>8}")
        print("-" * 72)
        for vn, wr_t, n_t, wr_f, n_f, lift in vote_imp:
            icon = "🟢" if lift > 0.05 else ("🔴" if lift < -0.05 else "➡️")
            print(f"  {vn:<26}{wr_t:>9.1%}{n_t:>6}{wr_f:>11.1%}{n_f:>6}{lift:>+7.1%} {icon}")

    # ── 11. Vote-combination breakdown (ported from v1) ──
    if args.vote_breakdown_range:
        try:
            lo_str, hi_str = args.vote_breakdown_range.split(",")
            lo, hi = float(lo_str), float(hi_str)
        except ValueError:
            sys.exit("--vote-breakdown-range must be 'LOW,HIGH', e.g. '24,25'")

        result = vote_combo_breakdown(rows, lo, hi, min_sample=args.min_sample)
        if result is None:
            print(f"\nNo rows with vote data in score range [{lo}, {hi}) "
                  f"(vote logging may not have been deployed yet when these fired).")
        else:
            band_rows, combo_stats = result
            print(f"\nVote-combination breakdown for score range [{lo}, {hi}) — {len(band_rows)} rows:")
            print("-" * 60)
            for combo, d in sorted(combo_stats.items(), key=lambda kv: -kv[1]["n"]):
                wr = d["wins"] / d["n"] if d["n"] else 0
                flag = "" if d["n"] >= args.min_sample else "  (low sample)"
                label = ", ".join(combo) if combo else "(no votes true)"
                print(f"  n={d['n']:<4} wr={wr:>6.1%}  {label}{flag}")

    # ── 12. FINAL RECOMMENDATION ──
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")

    # Target-WR floor: lowest cap where the WILSON LOWER BOUND (not raw WR)
    # clears the target with enough sample — statistically honest version
    # of "acceptable and not noise", per the v2.1 fix.
    target_floor = None
    for cap, n_pass, wr, wr_lo in caps_data:
        if n_pass >= args.min_sample and wr_lo >= args.target_winrate:
            target_floor = cap
            break
    # Fall back to raw-WR criterion if the stricter Wilson-bound version
    # finds nothing (keeps the tool usable on thin datasets).
    if target_floor is None:
        for cap, n_pass, wr, wr_lo in caps_data:
            if n_pass >= args.min_sample and wr >= args.target_winrate:
                target_floor = cap
                break

    knee_floor = knee if knee is not None else 0.0
    ev_floor = best_ev[0] if best_ev else 0.0

    # v2.1: toxic-zone ceiling is intentionally NOT included here — see
    # module docstring. It's surfaced as a warning against the final pick
    # instead of forcing the recommendation upward.
    recommended = max(knee_floor, ev_floor, target_floor or 0.0)

    rec_subset = [r for r in rows if r["score"] >= recommended]
    rec_n = len(rec_subset)
    rec_wr = sum(r["win"] for r in rec_subset) / rec_n if rec_n else 0
    rec_wins = [favourable_move(r) for r in rec_subset if r["win"]]
    rec_losses = [favourable_move(r) for r in rec_subset if not r["win"]]
    rec_avg_w = sum(rec_wins) / len(rec_wins) if rec_wins else 0
    rec_avg_l = sum(rec_losses) / len(rec_losses) if rec_losses else 0
    rec_ev = expected_value(len(rec_wins), rec_n - len(rec_wins), rec_avg_w, rec_avg_l)
    rec_rr = rr_ratio(rec_avg_w, rec_avg_l)

    # Guard against 0.0 or pathological recommendation
    if recommended <= 0.0 or not caps_data:
        print(f"\n  ❌ No statistically valid threshold found. Collect more data.")
        sys.exit(2)

    print(f"\n  ✅ Recommended CONFLUENCE_MIN_ABS_SCORE: {recommended:.1f}")
    print(f"     N={rec_n} | WR={rec_wr:.1%} | EV={rec_ev:+.3f}% per trade | R:R={format_rr(rec_rr)}")
    if rec_n < 50:
        print(f"     ⚠️  Only {rec_n} samples support this threshold — treat as provisional.")

    # Alert frequency impact
    total_alerts = n
    dropped = total_alerts - rec_n
    ts_list = [r["entry_ts"] for r in rows]
    weeks_of_data = (max(ts_list) - min(ts_list)) / (7 * 86400) if len(ts_list) > 1 else 0.1
    alerts_per_week_before = total_alerts / max(weeks_of_data, 0.1)
    alerts_per_week_after = rec_n / max(weeks_of_data, 0.1)
    print(f"     Alert frequency: {alerts_per_week_before:.1f}/week → {alerts_per_week_after:.1f}/week "
          f"(dropping {dropped} alerts, {dropped/total_alerts:.0%})")

    # Warn if the pick sits inside/below a toxic bucket range
    overlapping_toxic = [t for t in toxic_zones if t[0] < recommended < t[1] or
                          (recommended <= t[0])]
    if overlapping_toxic:
        worst = max(overlapping_toxic, key=lambda t: t[1])
        print(f"\n  ⚠️  Note: a toxic bucket ({worst[0]:.1f}-{worst[1]:.1f}, {worst[2]:.0%} WR) "
              f"exists at or above the recommended threshold.")
        print(f"      Cumulative stats above already price this in, but it's worth checking")
        print(f"      the per-alert breakdown to see what's firing in that band.")

    print(f"\n  Reasoning:")
    if knee is not None:
        print(f"     • Knee point at {knee:.1f} (diminishing returns beyond this, smoothed)")
    if best_ev:
        print(f"     • EV-optimal at {best_ev[0]:.1f} (maximises $ per trade)")
    if target_floor is not None:
        print(f"     • Target WR ({args.target_winrate:.0%}, Wilson lower bound) met at {target_floor:.1f}")
    if toxic_zones:
        toxic_ceiling = max(t[1] for t in toxic_zones)
        print(f"     • (Informational) toxic zones present up to {toxic_ceiling:.1f} — not used as a floor")
    print(f"     • Taking the MAX of knee / EV-optimal / target-WR floors → {recommended:.1f}")

    # Stability guidance
    print(f"\n  Stability: run this script weekly. If the recommendation")
    print(f"  moves by more than ±2.0 points between runs, the dataset")
    print(f"  is still too thin to trust a single number.")

    # Tiered options
    print(f"\n  Alternative tiers:")
    toxic_ceiling = max((t[1] for t in toxic_zones), default=0.0)
    for label, floor, desc in [
        ("Balanced", knee_floor, "Knee point — best WR/sample trade-off"),
        ("Sniper", ev_floor, "EV-optimal — maximises profit per trade"),
        ("Toxic-clear", toxic_ceiling, "Clears every flagged toxic bucket (informational, not forced)"),
    ]:
        sub = [r for r in rows if r["score"] >= floor]
        if sub:
            wr = sum(r["win"] for r in sub) / len(sub)
            print(f"     {label:<15} → {floor:.1f}  "
                  f"(N={len(sub)}, WR={wr:.1%}) — {desc}")

    print(f"\n{'='*70}")
    print(f"  To apply: set CONFLUENCE_MIN_ABS_SCORE = {recommended:.1f} in config")
    print(f"{'='*70}\n")

    # ── JSON output mode ──
    if args.json:
        output = {
            "recommended_score": recommended,
            "sample_size": rec_n,
            "win_rate": round(rec_wr, 4),
            "ev": round(rec_ev, 4),
            "rr": rec_rr,
            "direction_filter": args.direction,
            "pair_filter": args.pair,
            "alert_key_filter": args.alert_key,
            "tiers": {
                "balanced": {"score": knee_floor, "description": "Knee point"},
                "sniper": {"score": ev_floor, "description": "EV-optimal"},
                "toxic_clear": {"score": toxic_ceiling, "description": "Clears toxic buckets"},
            },
            "buy_wr": buy_wr,
            "buy_n": buy_n,
            "sell_wr": sell_wr,
            "sell_n": sell_n,
            "alerts_per_week_before": round(alerts_per_week_before, 2),
            "alerts_per_week_after": round(alerts_per_week_after, 2),
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
