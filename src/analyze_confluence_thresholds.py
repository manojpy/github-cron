#!/usr/bin/env python3
"""
analyze_confluence_thresholds.py  v3.0 — Intelligent Threshold Advisor (CLI)

Reads the outcome_log_stream Redis stream and prints a data-driven,
statistically-rigorous CONFLUENCE_MIN_ABS_SCORE recommendation.

"""

import argparse
import json
import os
import sys

try:
    import redis
except ImportError:
    sys.exit("Missing dependency: pip install redis")

import threshold_engine as engine

STREAM_KEY = "outcome_log_stream"



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
            "win": int(win),
            "pct_move": to_float(f.get("pct_move"), 0.0),
            "entry_ts": int(f.get("entry_ts", 0)),
            "votes": votes,
            "adx_val": to_float(f.get("adx_val")),
        })
    if not rows:
        print("No usable rows after filtering.", file=sys.stderr)
        sys.exit(2)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Intelligent Confluence Threshold Advisor v3.0")
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
    ap.add_argument("--walk-forward", action="store_true",
                    help="Fit the threshold on the older 2/3 of data, validate it against "
                         "the newer 1/3 it never saw, and report whether it holds up out-of-sample.")
    ap.add_argument("--train-frac", type=float, default=0.67,
                    help="Fraction of (chronologically) older rows used for fitting in --walk-forward mode")
    ap.add_argument("--monte-carlo", type=int, default=0, metavar="N_SIMS",
                    help="Run N block-bootstrap walk-forward simulations for a robustness distribution "
                         "instead of trusting a single split (e.g. --monte-carlo 100). Offline-only.")
    ap.add_argument("--regime", action="store_true",
                    help="Break down win rate by trending vs ranging regime (ADX-based, self-relative "
                         "median split). Diagnostic only — never changes the recommended threshold.")
    ap.add_argument("--attribution", metavar="WEIGHTS_JSON_PATH",
                    help="Counterfactual per-vote attribution: which votes are propping up otherwise-"
                         "weak trades over the threshold. Needs a JSON file of {vote_name: weight} — "
                         "e.g. dump cfg.CONFLUENCE_WEIGHTS from bot_config.py on the bot host, since "
                         "this script runs standalone without bot_config. Diagnostic only.")
    args = ap.parse_args()

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        sys.exit("Set REDIS_URL in your environment first.")

    r = redis.from_url(redis_url, decode_responses=True)
    rows = load_rows(r, args.pair, args.direction, args.alert_key)
    n = len(rows)
    overall_wr = sum(row["win"] for row in rows) / n

    buy_wr, buy_n, sell_wr, sell_n = engine.direction_split(rows)

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
    buckets = engine.build_buckets(rows, bw)
    avg_per_bucket = n / len(buckets) if buckets else 0

    print(f"{'Score bucket':<16}{'N':>6}{'Win rate':>12}{'95% CI':>18}{'Verdict':>12}")
    print("-" * 64)
    for b in sorted(buckets):
        d = buckets[b]
        wr = d["wins"] / d["n"] if d["n"] else 0
        lo, hi, _ = engine.wilson_ci(d["wins"], d["n"])
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

    # ── 2. Toxic zone detection (informational) ──
    toxic_zones = engine.detect_toxic_zones(buckets, bw, min_sample=args.min_sample)
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

    # ── 3. Anomalous buckets ──
    anomalies = engine.detect_anomalous_buckets(buckets, bw, min_sample=args.min_sample)
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
    print(f"\n{'Cap':>8}{'N':>8}{'WR':>10}{'95% CI':>18}{'EV/trade':>12}{'R:R':>8}")
    print("-" * 64)
    for cap in candidate_caps:
        subset = [row for row in rows if row["score"] >= cap]
        if not subset or len(subset) < 5:
            continue
        wins_count = sum(r["win"] for r in subset)
        wr = wins_count / len(subset)
        lo, hi, _ = engine.wilson_ci(wins_count, len(subset))
        ev, rr, _avg_w, _avg_l = engine.ev_and_rr_for(subset)
        print(f"  {cap:>6.1f}{len(subset):>8}{wr:>9.1%}"
              f"  [{lo:.0%}-{hi:.0%}]{ev:>+10.3f}%{engine.format_rr(rr):>8}")

    # ── Core recommendation — everything below is drawn from ONE call ──
    rec = engine.recommend_threshold(
        rows, target_winrate=args.target_winrate,
        min_sample=args.min_sample, bucket_size=bw,
    )

    knee = rec.get("knee")
    if knee is not None:
        knee_row = next((c for c in rec["caps_data"] if c[0] == knee), None)
        if knee_row:
            print(f"\n📐 Knee point detected at score {knee:.1f} "
                  f"(WR={knee_row[2]:.1%}, N={knee_row[1]})")
            print(f"   Beyond this point, each +1 score adds diminishing WR improvement.")

    best_ev = rec.get("best_ev")
    if best_ev:
        rr = engine.rr_ratio(best_ev[4], best_ev[5])
        print(f"\n💰 EV-optimal threshold: score >= {best_ev[0]:.1f}")
        print(f"   WR={best_ev[2]:.1%} | N={best_ev[1]} | "
              f"EV={best_ev[3]:+.3f}% per trade | R:R={engine.format_rr(rr)} "
              f"(avg win={best_ev[4]:+.2f}%, avg loss={best_ev[5]:+.2f}%)")
        if best_ev[3] <= 0:
            print(f"\n🚨 WARNING: EV-optimal threshold ({best_ev[0]:.1f}) has "
                  f"non-positive EV ({best_ev[3]:+.3f}%). The strategy may be "
                  f"unprofitable across all score levels.")

    recent_wr = rec.get("drift_recent_wr")
    if recent_wr is not None:
        older_wr = rec["drift_older_wr"]
        recent_n = rec["drift_recent_n"]
        drift = recent_wr - older_wr
        drift_icon = "📈" if drift > 0.02 else ("📉" if drift < -0.02 else "➡️")
        print(f"\n{drift_icon} Temporal drift: last 14d WR={recent_wr:.1%} ({recent_n} samples) "
              f"vs prior WR={older_wr:.1%} (Δ{drift:+.1%})")
        if drift < -0.05:
            print(f"   ⚠️  Edge may be decaying — consider raising threshold as a hedge.")

    # ── Per-pair breakdown ──
    pair_stats = engine.per_pair_breakdown(rows, min_sample=args.min_sample)
    if pair_stats:
        print(f"\n{'Pair':<12}{'WR':>8}{'N':>6}")
        print("-" * 28)
        for pair, wr, cnt in pair_stats:
            flag = " 🔴" if wr < 0.45 else (" 🟢" if wr >= 0.70 else "")
            print(f"  {pair:<10}{wr:>7.1%}{cnt:>6}{flag}")

    # ── Per-alert breakdown (worst 5 + best 5, no duplicates when <=10) ──
    alert_stats = engine.per_alert_breakdown(rows, min_sample=args.min_sample)
    if alert_stats:
        print(f"\n{'Alert key':<30}{'WR':>8}{'N':>6}{'Avg score':>12}")
        print("-" * 56)
        if len(alert_stats) <= 10:
            display = alert_stats
        else:
            display = alert_stats[:5] + alert_stats[-5:]
        for idx, (ak, wr, cnt, avg_s) in enumerate(display):
            if len(alert_stats) > 10 and idx == 5:
                print(f"  ... ({len(alert_stats) - 10} more) ...")
            flag = " 🔴" if wr < 0.45 else (" 🟢" if wr >= 0.70 else "")
            print(f"  {ak:<28}{wr:>7.1%}{cnt:>6}{avg_s:>11.1f}{flag}")

    # ── Vote importance ranking ──
    vote_imp = engine.vote_importance(rows, min_sample=args.min_sample)
    if vote_imp:
        print(f"\n{'Vote':<28}{'WR (True)':>10}{'N':>6}{'WR (False)':>12}{'N':>6}{'Lift':>8}")
        print("-" * 72)
        for vn, wr_t, n_t, wr_f, n_f, lift in vote_imp:
            icon = "🟢" if lift > 0.05 else ("🔴" if lift < -0.05 else "➡️")
            print(f"  {vn:<26}{wr_t:>9.1%}{n_t:>6}{wr_f:>11.1%}{n_f:>6}{lift:>+7.1%} {icon}")

    # ── Vote-combination breakdown ──
    if args.vote_breakdown_range:
        try:
            lo_str, hi_str = args.vote_breakdown_range.split(",")
            lo, hi = float(lo_str), float(hi_str)
        except ValueError:
            sys.exit("--vote-breakdown-range must be 'LOW,HIGH', e.g. '24,25'")

        result = engine.vote_combo_breakdown(rows, lo, hi, min_sample=args.min_sample)
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

    # ── FINAL RECOMMENDATION ──
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")

    if not rec["valid"]:
        error = rec.get("error")
        if error == "no_caps_data":
            print("\n  ❌ No cap levels have enough data. Cannot recommend a threshold.")
        else:
            print("\n  ❌ No statistically valid threshold found. Collect more data.")
        sys.exit(2)

    recommended = rec["recommended"]
    knee_floor = knee if knee is not None else 0.0
    ev_floor = best_ev[0] if best_ev else 0.0
    target_floor = rec.get("target_floor")

    print(f"\n  ✅ Recommended CONFLUENCE_MIN_ABS_SCORE: {recommended:.1f}")
    print(f"     N={rec['rec_n']} | WR={rec['rec_wr']:.1%} "
          f"[{rec['rec_wilson_lo']:.0%}-{rec['rec_wilson_hi']:.0%}] | "
          f"EV={rec['rec_ev']:+.3f}% per trade | R:R={engine.format_rr(rec['rec_rr'])}")
    print(f"     Confidence: {rec['confidence']}")
    if rec["rec_n"] < 50:
        print(f"     ⚠️  Only {rec['rec_n']} samples support this threshold — treat as provisional.")
    
    print(f"     Alert frequency: {rec['alerts_per_week_before']:.1f}/week → "
          f"{rec['alerts_per_week_after']:.1f}/week "
          f"(dropping {rec['dropped']} alerts, {rec['dropped_pct']:.0%})")

    if rec["overlapping_toxic"]:
        worst = max(rec["overlapping_toxic"], key=lambda t: t[1])
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
        print(f"     • (Informational) toxic zones present up to {rec['toxic_ceiling']:.1f} — not used as a floor")
    print(f"     • Taking the MAX of knee / EV-optimal / target-WR floors → {recommended:.1f}")

    print(f"\n  Stability: run this script weekly. If the recommendation")
    print(f"  moves by more than ±2.0 points between runs, the dataset")
    print(f"  is still too thin to trust a single number.")

    print(f"\n  Alternative tiers:")
    for label, floor, desc in [
        ("Balanced", knee_floor, "Knee point — best WR/sample trade-off"),
        ("Sniper", ev_floor, "EV-optimal — maximises profit per trade"),
        ("Toxic-clear", rec["toxic_ceiling"], "Clears every flagged toxic bucket (informational, not forced)"),
    ]:
        sub = [r for r in rows if r["score"] >= floor]
        if sub:
            wr = sum(r["win"] for r in sub) / len(sub)
            print(f"     {label:<15} → {floor:.1f}  "
                  f"(N={len(sub)}, WR={wr:.1%}) — {desc}")

    print(f"\n{'='*70}")
    print(f"  To apply: set CONFLUENCE_MIN_ABS_SCORE = {recommended:.1f} in config")
    print(f"{'='*70}\n")

    wf = None
    if args.walk_forward:
        wf = engine.validate_threshold_walk_forward(
            rows, target_winrate=args.target_winrate, min_sample=args.min_sample,
            bucket_size=bw, train_frac=args.train_frac,
        )
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD VALIDATION (train={args.train_frac:.0%} of older rows, "
              f"holdout=rest)")
        print(f"{'='*70}")
        if not wf["valid"]:
            print(f"\n  ❌ Could not validate: {wf.get('error')} "
                  f"(train_n={wf.get('train_n', 0)}, holdout_n={wf.get('holdout_n', 0)})")
            print(f"     Need at least {args.min_sample * 2} train rows and "
                  f"{args.min_sample} holdout rows. Collect more data first.")
        else:
            print(f"\n  Train set:   {wf['train_n']} rows → threshold {wf['recommended']:.1f} "
                  f"(train WR {wf['train_result']['rec_wr']:.1%}, N={wf['train_result']['rec_n']})")
            if wf.get("passed") is None:
                print(f"  Holdout set: {wf['holdout_n']} rows, but only "
                      f"{wf['holdout_n_at_threshold']} clear score >= {wf['recommended']:.1f} "
                      f"— too thin to judge. Collect more data.")
            else:
                icon = "✅ PASSED" if wf["passed"] else "🚨 FAILED"
                print(f"  Holdout set: {wf['holdout_n']} rows, {wf['holdout_n_at_threshold']} "
                      f"clear the threshold → OOS WR {wf['holdout_wr']:.1%} "
                      f"[{wf['holdout_wilson_lo']:.0%}-{wf['holdout_wilson_hi']:.0%}]")
                print(f"  {icon} — degraded {wf['degraded_pct']:+.1%} vs train WR "
                      f"(target {args.target_winrate:.0%}, slack 5pt)")

                if not wf["passed"]:
                    print(f"     This threshold looked good on the data it was chosen from but")
                    print(f"     didn't hold up on newer data it hadn't seen. Treat the")
                    print(f"     recommendation above as unconfirmed — don't apply it yet.")

    mc = None
    if args.monte_carlo > 0:
        mc = engine.monte_carlo_walk_forward(
            rows, n_simulations=args.monte_carlo, train_frac=args.train_frac,
            min_sample=args.min_sample, target_winrate=args.target_winrate, bucket_size=bw,
        )
        print(f"\n{'='*70}")
        print(f"  MONTE CARLO ROBUSTNESS ({args.monte_carlo} block-bootstrap simulations)")
        print(f"{'='*70}")
        if not mc["valid"]:
            print(f"\n  ❌ Could not run: {mc.get('error')} — collect more data first.")
        else:
            print(f"\n  {mc['n_simulations']}/{mc['n_requested']} simulations produced a valid result.")
            print(f"  OOS win rate:  mean {mc['oos_wr_mean']:.1%}  ±{mc['oos_wr_std']:.1%}  "
                  f"[p5={mc['oos_wr_p5']:.1%}, p95={mc['oos_wr_p95']:.1%}]")
            print(f"  Threshold:     mean {mc['threshold_mean']:.1f}  ±{mc['threshold_std']:.1f}")

            icon = "✅ ROBUST" if mc["robustness_score"] > 2.0 else "⚠️ FRAGILE — needs more data"
            print(f"  Robustness score: {mc['robustness_score']:.2f}  {icon}")
            print(f"  (p5 is the number to trust for a worst-case plan, not the mean)")

    rb = None
    if args.regime:
        rb = engine.regime_breakdown(rows, min_sample=args.min_sample)
        print(f"\n{'='*70}")
        print(f"  REGIME BREAKDOWN (trending vs ranging, ADX self-relative median split)")
        print(f"{'='*70}")
        if not rb["valid"]:
            print(f"\n  ❌ Could not break down: {rb.get('error')} "
                  f"({rb['n_with_adx']}/{rb['n_total']} rows have adx_val — "
                  f"older outcome rows predate this field and are skipped).")
        else:
            print(f"\n  Median ADX in this window: {rb['median_adx']:.1f}")
            for label in ("trending", "ranging"):
                d = rb["regimes"].get(label, {})
                if not d.get("valid"):
                    print(f"  {label:<10} — {d.get('error', 'no data')} (n={d.get('n', 0)})")
                    continue
                print(f"  {label:<10} n={d['n']:<5} WR={d['wr']:.1%} "
                      f"[{d['wilson_lo']:.0%}-{d['wilson_hi']:.0%}]  confidence={d['confidence']}")
            if "wr_gap" in rb:
                gap = rb["wr_gap"]
                print(f"\n  Gap (trending − ranging): {gap:+.1%}")
                if abs(gap) > 0.10:
                    print(f"  Meaningful gap — this edge is NOT regime-neutral. Worth tracking")
                    print(f"  separately before ever picking a regime-specific multiplier.")
                else:
                    print(f"  Small gap — this edge looks roughly regime-neutral so far.")

    attribution = None
    if args.attribution:
        try:
            with open(args.attribution) as fh:
                weights = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"\n  ❌ Could not read weights file {args.attribution}: {e}", file=sys.stderr)
            weights = None
        if weights:
            attribution = engine.outcome_attribution(rows, weights, threshold=recommended, min_sample=args.min_sample)
            print(f"\n{'='*70}")
            print(f"  OUTCOME ATTRIBUTION (counterfactual, at threshold {recommended:.1f})")
            print(f"{'='*70}")
            if not attribution:
                print(f"\n  No votes had enough rescued-trade samples to evaluate yet.")
            else:
                for e in attribution:
                    if not e["rescued_valid"]:
                        print(f"  {e['vote']:<20} rescues {e['n_rescued']} — too thin to judge yet")
                        continue
                    flag = "⚠️ " if e["rescued_wr"] < args.target_winrate - 0.10 else "   "
                    print(f"  {flag}{e['vote']:<18} rescued {e['n_rescued']:>4} "
                          f"({e['rescued_pct']:.0%} of its True cases) → WR {e['rescued_wr']:.1%} "
                          f"[{e['rescued_wilson_lo']:.0%}-{e['rescued_wilson_hi']:.0%}] {e['rescued_confidence']}")
                print(f"\n  Rescued = trades that only cleared the threshold because of that vote's")
                print(f"  weight. Low WR among a vote's rescued trades means it's propping up")
                print(f"  otherwise-weak setups over the line, not adding real edge.")

    if args.json:
        output = {
            "recommended_score": recommended,
            "sample_size": rec["rec_n"],
            "win_rate": round(rec["rec_wr"], 4),
            "ev": round(rec["rec_ev"], 4),
            "rr": rec["rec_rr"],
            "direction_filter": args.direction,
            "pair_filter": args.pair,
            "alert_key_filter": args.alert_key,
            "tiers": {
                "balanced": {"score": knee_floor, "description": "Knee point"},
                "sniper": {"score": ev_floor, "description": "EV-optimal"},
                "toxic_clear": {"score": rec["toxic_ceiling"], "description": "Clears toxic buckets"},
            },
            "buy_wr": rec.get("buy_wr"),
            "buy_n": rec.get("buy_n"),
            "sell_wr": rec.get("sell_wr"),
            "sell_n": rec.get("sell_n"),
            "alerts_per_week_before": round(rec["alerts_per_week_before"], 2),
            "alerts_per_week_after": round(rec["alerts_per_week_after"], 2),
        }
        if wf is not None:
            output["walk_forward"] = {
                "valid": wf["valid"],
                "passed": wf.get("passed"),
                "error": wf.get("error"),
                "train_n": wf.get("train_n"),
                "holdout_n": wf.get("holdout_n"),
                "holdout_n_at_threshold": wf.get("holdout_n_at_threshold"),
                "holdout_wr": round(wf["holdout_wr"], 4) if wf.get("holdout_wr") is not None else None,
            }
        if mc is not None:
            output["monte_carlo"] = {
                "valid": mc["valid"],
                "error": mc.get("error"),
                "n_simulations": mc.get("n_simulations"),
                "oos_wr_mean": round(mc["oos_wr_mean"], 4) if mc.get("oos_wr_mean") is not None else None,
                "oos_wr_std": round(mc["oos_wr_std"], 4) if mc.get("oos_wr_std") is not None else None,
                "oos_wr_p5": round(mc["oos_wr_p5"], 4) if mc.get("oos_wr_p5") is not None else None,
                "oos_wr_p95": round(mc["oos_wr_p95"], 4) if mc.get("oos_wr_p95") is not None else None,
                "robustness_score": round(mc["robustness_score"], 3) if mc.get("robustness_score") is not None else None,
            }
        if rb is not None:
            output["regime_breakdown"] = {
                "valid": rb["valid"],
                "error": rb.get("error"),
                "median_adx": round(rb["median_adx"], 2) if rb.get("median_adx") is not None else None,
                "wr_gap": round(rb["wr_gap"], 4) if rb.get("wr_gap") is not None else None,
                "regimes": {
                    label: {
                        "valid": d.get("valid", False),
                        "n": d.get("n"),
                        "wr": round(d["wr"], 4) if d.get("wr") is not None else None,
                        "confidence": d.get("confidence"),
                    }
                    for label, d in rb.get("regimes", {}).items() 
                } if rb["valid"] else {},
            }
        if attribution is not None:
            output["outcome_attribution"] = [
                {
                    "vote": e["vote"], "weight": e["weight"],
                    "n_with_vote": e["n_with_vote"], "n_rescued": e["n_rescued"],
                    "rescued_pct": round(e["rescued_pct"], 4),
                    "rescued_valid": e["rescued_valid"],
                    "rescued_wr": round(e["rescued_wr"], 4) if e.get("rescued_wr") is not None else None,
                }
                for e in attribution
            ]
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
