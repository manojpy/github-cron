# 🤖 MACD Unified Bot

High-performance cryptocurrency trading alert bot with AOT/Cython/Numba compilation, Redis state management, outcome tracking, and Telegram notifications. Runs on GitHub Actions, triggered every 15 minutes by an external scheduler (Cronjobs.org).

**Version**: 1.8.x | **Python**: 3.11 | **Last audited**: 2026-10-10

---

## 📋 Quick Overview

| Aspect | Detail |
|--------|--------|
| **What** | Analyzes crypto pairs with 20+ technical indicators and confluence gates |
| **When** | Every 15 minutes at :01, :16, :31, :46, triggered externally by Cronjobs.org |
| **Outputs** | Telegram alerts with smart deduplication + optional Brain reports |
| **Speed** | Typically 15–45 s for a full cycle (AOT path) |
| **Memory** | Soft limit ~850 MB, container hard limit 900 MB |
| **State** | Redis (dedup, locks, stats, config overrides) + file-based outcome archive |

---

## 🏭 Production Context (important)

- The main bot (`run-bot.yml` → `macd_unified.py`) is triggered by an **external Cronjobs.org job**, not by a GitHub Actions `schedule`.
- **Schedule**: every 15 minutes at minutes **1, 16, 31, 46** of every hour (`:01`, `:16`, `:31`, `:46`), one minute after each 15 m candle close.
- **Purpose of each run**: fetch market data, evaluate signals, and send Telegram alerts when conditions are met.
- **Redis** is the shared state store. **Telegram** is the only outbound notification channel.
- The bot is safe to run frequently: no duplicate alerts, no wasted API calls, no unbounded Redis growth, and minimal round-trips.

### What triggers what

| Workflow | File | Trigger | Cadence |
|----------|------|---------|---------|
| Run MACD Unified Bot | `run-bot.yml` | Cronjobs.org → `workflow_dispatch` | Every 15 min (:01, :16, :31, :46) |
| Watchdog | `watchdog.yml` | Cronjobs.org → `workflow_dispatch` | Every 30 min |
| Learner | `learner.yml` | Cronjobs.org → `workflow_dispatch` | Every 6 hours |
| Cleanup outcomes | `cleanup-outcomes.yml` | Cronjobs.org → `workflow_dispatch` | Daily at 02:00 (timezone as set on the Cronjobs.org job) |
| Redis audit | `redis-audit.yml` | `workflow_run` after the Learner completes | Chained, once per Learner run |
| Replay un-pushed outcomes | `replay-outcomes.yml` | `workflow_run` after the bot run completes | Does work only after a non-successful bot run |
| Build AOT image | `build.yml` | Push to `main` (src / Dockerfile / requirements) + weekly GitHub `schedule` (`0 2 * * 0`, Sunday 02:00 UTC) + manual | On code change and weekly |
| CI | `ci.yml` | Push, pull request, manual | On code change |

`build.yml` is the only workflow with a GitHub `schedule`. Learner, Watchdog and Cleanup have **no** GitHub schedule, so if Cronjobs.org stops calling them nothing else will.

### Overlap and ordering

`run-bot.yml` uses a concurrency group with `cancel-in-progress: false`. If a run is still going when the next trigger arrives, the new run waits instead of cancelling it, and GitHub keeps at most one pending run per group. The bot also takes a Redis lock (`macd_bot_run`, 600 s) so two bot processes never evaluate at the same time.

---

## 🚀 Setup (5 Steps)

### 1. Fork & Configure Secrets

Add these in **Settings → Secrets and variables → Actions**:

```
TELEGRAM_BOT_TOKEN     → from @BotFather
TELEGRAM_CHAT_ID       → your chat / group ID
REDIS_URL              → redis://user:pass@host:port  (or rediss://)
DELTA_API_BASE         → https://api.india.delta.exchange
DATA_REPO_TOKEN        → PAT with write access to the outcome-data repo
```

### 2. Edit Configuration

```bash
# Edit the checked-in config
nano config_macd.json
```

Key settings (see full file for 80+ options):

```json
{
  "PAIRS": ["BTCUSD", "ETHUSD", "..."],   // currently 30 pairs
  "MAX_PARALLEL_FETCH": 12,
  "EVAL_CONCURRENCY_LIMIT": 4,
  "RUN_TIMEOUT_SECONDS": 480,
  "MEMORY_LIMIT_BYTES": 850000000,
  "FAIL_ON_REDIS_DOWN": true,
  "FAIL_ON_TELEGRAM_DOWN": false,
  "DRY_RUN_MODE": false,
  "ENABLE_BRAIN": true,
  "BRAIN_SHADOW_MODE": true
}
```

### 3. Push & Build

```bash
git add config_macd.json
git commit -m "Configure bot"
git push
```

This triggers `build.yml` → multi-stage Docker image with Cython + AOT compilation → push to `ghcr.io`.

### 4. Verify Build

- Actions tab → **Build AOT Image**
- Wait for ✅ (usually 3–6 minutes)

### 5. Set Up the External Triggers

Create one Cronjobs.org job per workflow in the table above. Each job calls the GitHub API `workflow_dispatch` endpoint for that workflow file on the `main` branch, using a token that is allowed to dispatch workflows. For the main bot use the cron expression `1,16,31,46 * * * *`.

To test without the scheduler: Actions → **Run MACD Unified Bot** → Run workflow. Optional inputs: dry-run, Brain report, apply Brain plan, clear Redis, clear kill-switch.

Results appear in Telegram and in the workflow summary / artifacts.

---

## ⏰ Scheduling

Only `workflow_dispatch` is declared in `run-bot.yml`, `watchdog.yml`, `learner.yml` and `cleanup-outcomes.yml`. This is intentional: timing comes from Cronjobs.org, which is more punctual than GitHub's own cron queue. Do not add a `schedule:` block on top of it, or you will get double runs.

The bot run takes `TRIGGER_TIMESTAMP` (set when the workflow step starts) as its reference time and logs it in IST. If that timestamp is more than 10 minutes away from the current time, the bot falls back to the current time.

The **Watchdog** (`watchdog.yml`, called every 30 minutes) queries the last runs of `run-bot.yml` and sends a Telegram alert when:

- no successful run has happened for **40 minutes or more** (first alert at 40–69 min, then roughly every 2 hours), or
- the latest run finished with a conclusion other than success, cancelled or skipped.

If the Watchdog itself stops being called, there is no alert. Check the Cronjobs.org execution history first when things go quiet.

---

## ⚙️ Configuration Quick Reference

Values below match the checked-in `config_macd.json`.

```json
{
  // REQUIRED (from GitHub Secrets — never commit real values)
  "TELEGRAM_BOT_TOKEN": "...",
  "TELEGRAM_CHAT_ID": "...",
  "REDIS_URL": "...",
  "DELTA_API_BASE": "https://api.india.delta.exchange",

  // Pairs (30 today; keep conservative relative to 900 MB / 2 CPU container)
  "PAIRS": ["BTCUSD", "ETHUSD", "..."],

  // Performance & limits
  "MAX_PARALLEL_FETCH": 12,
  "EVAL_CONCURRENCY_LIMIT": 4,
  "HTTP_TIMEOUT": 10,
  "RUN_TIMEOUT_SECONDS": 480,
  "FETCH_PHASE_TIMEOUT_SEC": 60,
  "MEMORY_LIMIT_BYTES": 850000000,
  "MAX_ALERTS_PER_PAIR": 9,
  "MAX_ALERTS_PER_RUN": 50,

  // Redis
  "REDIS_LOCK_EXPIRY": 600,
  "STATE_EXPIRY_DAYS": 11,
  "ALERT_DEDUP_WINDOW_SEC": 120,
  "COALESCE_DEDUP_WINDOW_SEC": 840,
  "FAIL_ON_REDIS_DOWN": true,

  // Resilience
  "FAIL_ON_TELEGRAM_DOWN": false,
  "MAX_CANDLE_STALENESS_SEC": 1200,
  "CANDLE_MIN_AGE_BUFFER": 45,

  // Brain / outcomes
  "ENABLE_BRAIN": true,
  "BRAIN_SHADOW_MODE": true,
  "BRAIN_AUTO_APPLY_DYNAMIC_WEIGHTS": false,
  "BRAIN_AUTO_DISABLE_ENABLED": true,
  "OUTCOME_PRIMARY_METRIC": "mfe",
  "OUTCOME_LOOKAHEAD_CANDLES": 12,
  "ENABLE_SINGLE_ACTIVE_TRADE": true,
  "TRADE_CLOSE_COOLDOWN_CANDLES": 3
}
```

Full option list lives in `config_macd.json` and is validated at startup by `bot_config.py`.

---

## 📊 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.11 | Core logic |
| Compilation | Numba (JIT + AOT) + Cython | Fast indicator path |
| Async | asyncio + aiohttp | Concurrent fetches & evaluation |
| State | Redis / Valkey | Dedup, locks, stats, overrides |
| Outcomes | JSONL files in separate repo | Brain analysis archive |
| Notifications | Telegram Bot API | Alert delivery |
| Deployment | Docker + GitHub Actions + GHCR, externally triggered | Build & runs |
| Container | Non-root, read-only rootfs | 900 MB memory limit, 2 CPUs |

---

## 📈 Indicators & Signals (summary)

**Indicators** (Numba/AOT/Cython accelerated):
EMA / RMA / SMA, PPO, RSI / Smoothed RSI, VWAP, Kalman & Range filters, MMH, Ichimoku Cloud, ATR/ADX adaptive, volume/RVOL, pivots/CPR, dynamic flow, etc.

**Alert families** (gated by confluence + many quality checks):
PPO crosses, RSI crosses, VWAP, pivots (P/R1–R3/S1–S3), MMH reversals, cloud/CHOCH/fib/strong-reversal, and more.
Alerts carry IST timestamp, price, key indicator values, wick quality, and confluence score.

Deduplication uses Redis (short window + optional coalescing). Candle non-repaint confirmation and mark-price agreement checks can release or keep the dedup claim.

---

## 🧠 Brain & Outcomes

- Real outcomes and shadow outcomes are written as daily JSONL files.
- A separate data repository (`outcome-data`) is sparse-checked out (3 days for normal runs, 185 days for Brain-report runs).
- Brain can emit reports, optionally auto-apply dynamic confluence weights (off in the current config), and auto-disable under-performing alert keys.
- Kill-switch logic exists but is off by default in the current config.
- The daily Cleanup workflow prunes old archives / reports with size and age limits.
- The Learner (every 6 hours) rebuilds the validated trade-plan playbook; the Redis audit runs right after it and checks key families and TTLs.

### How an alert is labelled

Every alert that passes the signal gates is still sent to Telegram. What differs is how it is tracked afterwards.

| Label | Meaning |
|-------|---------|
| 📝 **Recorded** | Counted as a trade. One recorded trade per pair per candle (the strongest edge). |
| ⏭ **Ignored** | Sent, but not recorded because the pair already has an open recorded trade (`ENABLE_SINGLE_ACTIVE_TRADE`). Log line: `NOT RECORDED … trade already open`. |
| ⏸ **Cooldown** | Sent, but not recorded or shadowed because a recorded target was hit within the last `TRADE_CLOSE_COOLDOWN_CANDLES` candles. |
| 👁 **Shadowed** | Alert was blocked by a gate (confluence, cluster penalty, win-rate, calibration, OOD, portfolio heat, brain-disabled). It is not sent, but it is tracked as a counterfactual trade for the Brain. |

Recorded and Shadowed trades keep **separate cooldowns** (`trade_cooldown:{pair}` and `trade_cooldown:shadow:{pair}`), so a counterfactual target hit never puts the live pair into cooldown.

### Reading the totals line

```
📝 Recorded - 53, Target Achieved - 18, Stop loss Hit - 35, Win Rate - 34%
👁 Shadowed - 18, Target Achieved - 13, Stop loss Hit - 5, Win Rate - 72%
```

This line is logged on every run (every 15 minutes) and is also included in the Brain report, which is sent every 12 hours (00:00 and 12:00 UTC). Both use the same formatter (`format_outcome_totals` in `state.py`).

- These are **all-time** totals of completed trades, not a rolling 24-hour counter.
- A trade counts as Target Achieved if its target is hit before its stop, and as Stop loss Hit if the stop comes first. If neither is hit within `OUTCOME_LOOKAHEAD_CANDLES` (12 candles = 3 h), it is counted by whether it closed in profit, so those timeouts are folded into the two counts.
- Trades are counted as soon as their target or stop is hit, even before the 12 candles are over. A `<Recorded|Shadowed> trade closed | …` line is logged at that moment.
- A new shadow registration does not change the total until it resolves, which can take up to 12 candles. `Pre-scanned N shadow pending outcome(s)` at run start shows how many are still open.
- Shadow rows record every blocked alert key, including several on the same pair and candle. They do not follow the one-trade-per-pair rule, so Shadowed and Recorded win rates are not directly comparable.
- Several shadow gates only activate once enough history exists: the win-rate filter needs `MIN_WIN_RATE_SAMPLE` resolved trades per pair and alert key, calibration needs curves from past outcomes, and the OOD gate needs `OOD_MIN_HISTORY` vote-count samples. Until then the confluence gate is the main source of shadow trades.

---

## 🔧 Local Development

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="src:$PYTHONPATH"
python src/macd_unified.py --validate-only   # config check
python src/macd_unified.py --debug           # full run with debug logs
```

Docker test (requires secrets and a config):

```bash
docker build -t macd-local .
docker run --rm \
  -e TELEGRAM_BOT_TOKEN="..." \
  -e TELEGRAM_CHAT_ID="..." \
  -e REDIS_URL="..." \
  -e DELTA_API_BASE="https://api.india.delta.exchange" \
  -v $(pwd)/config_macd.json:/app/src/config_macd.json:ro \
  macd-local
```

---

## 🐛 Troubleshooting

### Bot never runs on its own
- Open the Cronjobs.org job for the bot and check its execution history and last HTTP response (a failing GitHub token shows up there).
- Confirm the cron expression is `1,16,31,46 * * * *` and that the job targets the `main` branch.
- Check Telegram for Watchdog silence alerts. If the Watchdog is silent too, check its own Cronjobs.org job.

### Learner / Redis audit / Cleanup did not run
- Learner, Watchdog and Cleanup depend entirely on their Cronjobs.org jobs. There is no GitHub fallback schedule.
- The Redis audit only runs after a Learner run completes. If the Learner did not run, the audit did not either.

### Redis connection / quota / OOM
```
❌ REDIS_URL format or auth wrong
❌ "max requests limit exceeded" or "OOM command not allowed"
✅ Test: redis-cli -u "$REDIS_URL" ping
✅ On quota/OOM the bot stays degraded for the rest of the run (dedup disabled).
✅ Prefer a plan with adequate request & memory headroom for 30 pairs.
```

### Circuit breaker OPENED
```
❌ Delta API returning repeated 5xx / network errors
✅ Auto-recovers after recovery timeout (default 60 s)
```

### Memory limit exceeded / timeout
```
❌ Too many pairs or heavy Brain full-archive run
✅ Reduce PAIRS or split bots
✅ Check container logs for RSS vs MEMORY_LIMIT_BYTES
```

### Candle staleness / unstable candle
```
❌ Data older than MAX_CANDLE_STALENESS_SEC or too soon after close
✅ Workflow already warns when AGE < CANDLE_MIN_AGE_BUFFER or > staleness
✅ Increase buffer or staleness only if you understand the risk
```

### Duplicate alerts
```
❌ Redis degraded (quota/OOM) → dedup intentionally skipped
❌ Dedup window too short for your alert volume
✅ Inspect Redis keys matching recent_alert:* and pair_state:*
```

### Outcomes not persisted
```
❌ DATA_REPO_TOKEN missing or insufficient permissions
❌ Concurrent push from cleanup workflow exhausted rebase retries
✅ Check the "Persist outcomes" step logs and git status output
✅ After a failed run, replay-outcomes.yml recovers the unpushed-outcomes-* artifact
```

### Shadowed total not moving
```
❌ No gate blocked an alert (nothing to shadow), or shadows are still pending
✅ Look for "Shadow pending created" in the run log (one line per registration)
✅ Look for "Confluence gate blocked" lines: each should be followed by a registration
✅ "Cooldown after target … not shadowing" means the shadow cooldown suppressed it
```

---

## 📁 Project Structure

```
github-cron/
├── src/
│   ├── macd_unified.py          # Main entry & orchestration
│   ├── bot_config.py            # Config loading & validation
│   ├── fetcher.py               # HTTP, rate limit, circuit breaker, caches
│   ├── indicators.py            # Indicator helpers
│   ├── gates.py                 # Confluence / quality gates
│   ├── alerts.py                # Evaluation, formatting, Telegram queue
│   ├── state.py                 # Redis store, locks, pipelines, quota handling
│   ├── threshold_engine.py      # Calibration & thresholds
│   ├── brain*.py                # Brain analysis, shadow, audit, repair
│   ├── learner.py / playbook.py # Learner and trade-plan playbook
│   ├── outcome_storage.py       # JSONL outcome writers
│   ├── archive_reader.py        # Schema-aware outcome reading
│   ├── redis_audit.py           # Key / TTL audit
│   ├── aot_bridge.py / aot_meta.py / numba_functions_shared.py
│   └── cython_functions.pyx
├── .github/workflows/
│   ├── build.yml                # Docker + AOT/Cython image → GHCR (push + weekly)
│   ├── run-bot.yml              # Main bot execution (dispatched every 15 min)
│   ├── watchdog.yml             # Silence / failure watchdog (dispatched every 30 min)
│   ├── learner.yml              # Playbook learner (dispatched every 6 h)
│   ├── redis-audit.yml          # Key / TTL audit (chained after learner)
│   ├── cleanup-outcomes.yml     # Daily archive pruning (dispatched 02:00)
│   ├── replay-outcomes.yml      # Recovers un-pushed outcomes (chained after bot run)
│   └── ci.yml                   # Syntax & basic tests
├── config_macd.json
├── Dockerfile                   # Multi-stage, non-root, 900 MB limit
├── requirements.txt
└── tests/
```

---

## 🎯 Runtime Architecture (simplified)

```
Cronjobs.org (:01 :16 :31 :46)
        │  workflow_dispatch
        ▼
run-bot.yml
  • sparse-checkout config
  • pull GHCR image
  • decide Brain archive depth (3 d vs 185 d)
  • sparse-clone outcome-data repo
  • docker run (2 CPU, 900 MB, 660 s outer timeout)
        │
        ▼
macd_unified.py
  • connect Redis (with quota/OOM detection), take run lock
  • optional CLEAR_REDIS / CLEAR_KILL_SWITCH
  • parallel candle fetch (15 m / 5 m / daily)
  • indicator calc (AOT path preferred)
  • gate + alert evaluation
  • Redis pipelines for state / dedup / stats
  • Telegram queue (coalesced / batched)
  • write outcomes → mounted data repo
        │
        ▼
Persist step (rebase + retry push to outcome-data)
        │
        ▼ (on a non-successful run)
replay-outcomes.yml → recovers the unpushed-outcomes artifact
```

---

## 🔐 Security Notes

- Secrets live only in GitHub Secrets / environment; never in the repo.
- The Cronjobs.org job holds a token that can dispatch workflows. Scope it to this repository and the Actions permission only.
- Container runs as non-root (`appuser`), read-only root filesystem, limited tmpfs.
- Redis and Telegram URLs/tokens are redacted in normal logging paths.
- Outcome data repo access is token-scoped.

---

## 📈 Monitoring Checklist

1. Watchdog Telegram alerts (silence or failed conclusion).
2. Cronjobs.org execution history for all four dispatched jobs.
3. Workflow summary: duration, alerts sent, pairs scanned, memory, Redis status.
4. Artifacts: `bot-execution-logs-*` (7-day retention).
5. Redis: `KEYS pair_state:*`, `SCAN … MATCH recent_alert:*`, memory / command stats.
6. Redis audit output after each Learner run (missing TTLs, unexpected key families).
7. Outcome-data repo: daily JSONL growth and Brain reports under `reports/`.

---

## 🤝 Support

- Open issues with redacted logs + relevant config snippets.
- Prefer the workflow summary and uploaded log artifact when reporting failures.
- PRs welcome for bug fixes, clearer limits, and additional tests.

---

**Resources**

- [Numba](https://numba.readthedocs.io/)
- [Delta Exchange API](https://api.india.delta.exchange/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Redis / Valkey](https://redis.io/docs/)
