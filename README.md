# 🤖 MACD Unified Bot

High-performance cryptocurrency trading alert bot with AOT/Cython/Numba compilation, Redis state management, outcome tracking, and Telegram notifications. Designed to run on GitHub Actions every 15 minutes.

**Version**: 1.8.x | **Python**: 3.11 | **Last audited**: 2026-09-27

---

## 📋 Quick Overview

| Aspect | Detail |
|--------|--------|
| **What** | Analyzes crypto pairs with 20+ technical indicators and confluence gates |
| **When** | Intended every 15 minutes (1, 16, 31, 46 past the hour) via GitHub Actions |
| **Outputs** | Telegram alerts with smart deduplication + optional Brain reports |
| **Speed** | Typically 25–45 s for a full cycle (AOT path) |
| **Memory** | Soft limit ~850 MB, container hard limit 900 MB |
| **State** | Redis (dedup, locks, stats, config overrides) + file-based outcome archive |

> **Important**: As of the latest code, `run-bot.yml` only declares `workflow_dispatch`.  
> A `schedule` cron is **not** present in the workflow file.  
> You must either add the cron block (recommended) or trigger the workflow externally at the desired times.  
> The watchdog expects successful runs roughly every 15 minutes.

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
  "PAIRS": ["BTCUSD", "ETHUSD", "..."],   // currently ~30 pairs — consider ≤15–18 for safety
  "MAX_PARALLEL_FETCH": 12,
  "EVAL_CONCURRENCY_LIMIT": 4,
  "RUN_TIMEOUT_SECONDS": 480,
  "MEMORY_LIMIT_BYTES": 850000000,
  "FAIL_ON_REDIS_DOWN": false,
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

### 5. Run the Bot

- **Recommended**: Add a schedule to `run-bot.yml` (see below) so it runs automatically.  
- **Manual**: Actions → **Run MACD Unified Bot** → Run workflow.  
- Optional inputs: dry-run, Brain report, apply Brain plan, clear Redis, clear kill-switch.

Results appear in Telegram and in the workflow summary / artifacts.

---

## ⏰ Scheduling (Critical)

The workflow currently has **only** `workflow_dispatch`. To make it run on the 15-minute cadence the rest of the system expects, add:

```yaml
on:
  schedule:
    - cron: "1,16,31,46 * * * *"   # 1 minute after each 15 m candle close
  workflow_dispatch:
    # ... existing inputs ...
```

Keep the existing concurrency block:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: false
```

A separate **Watchdog** workflow (`watchdog.yml`) runs every 30 minutes and alerts on Telegram if no successful bot run has occurred for >40 minutes.

---

## ⚙️ Configuration Quick Reference

```json
{
  // REQUIRED (from GitHub Secrets — never commit real values)
  "TELEGRAM_BOT_TOKEN": "...",
  "TELEGRAM_CHAT_ID": "...",
  "REDIS_URL": "...",
  "DELTA_API_BASE": "https://api.india.delta.exchange",

  // Pairs (keep conservative relative to 900 MB / 2 CPU container)
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
  "REDIS_LOCK_EXPIRY": 900,
  "STATE_EXPIRY_DAYS": 11,
  "ALERT_DEDUP_WINDOW_SEC": 120,
  "COALESCE_DEDUP_WINDOW_SEC": 900,
  "FAIL_ON_REDIS_DOWN": false,

  // Resilience
  "FAIL_ON_TELEGRAM_DOWN": false,
  "MAX_CANDLE_STALENESS_SEC": 1200,

  // Brain / outcomes
  "ENABLE_BRAIN": true,
  "BRAIN_SHADOW_MODE": true,
  "BRAIN_AUTO_APPLY_DYNAMIC_WEIGHTS": true,
  "BRAIN_AUTO_DISABLE_ENABLED": true,
  "OUTCOME_PRIMARY_METRIC": "net_pnl_pct"
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
| Deployment | Docker + GitHub Actions + GHCR | Build & scheduled/manual runs |
| Container | Non-root, read-only rootfs | 900 MB memory limit |

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
- Brain can emit reports, auto-apply dynamic confluence weights, and (when enabled) auto-disable under-performing alert keys.
- Kill-switch logic exists but is off by default in the current config.
- Cleanup workflow runs daily and prunes old archives / reports with size and age limits.

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
- Confirm a `schedule` cron exists in `run-bot.yml` **or** that an external system is calling `workflow_dispatch` at the expected times.
- Check the Watchdog workflow for silence alerts.

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
✅ Reduce PAIRS (recommended ≤15–18) or split bots
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
│   ├── outcome_storage.py       # JSONL outcome writers
│   ├── archive_reader.py        # Schema-aware outcome reading
│   ├── aot_bridge.py / aot_meta.py / numba_functions_shared.py
│   └── cython_functions.pyx
├── .github/workflows/
│   ├── build.yml                # Docker + AOT/Cython image → GHCR
│   ├── run-bot.yml              # Main bot execution (currently dispatch-only)
│   ├── watchdog.yml             # Silence / failure watchdog
│   ├── cleanup-outcomes.yml     # Daily archive pruning
│   └── ci.yml                   # Syntax & basic tests
├── config_macd.json
├── Dockerfile                   # Multi-stage, non-root, 900 MB limit
├── requirements.txt
└── tests/
```

---

## 🎯 Runtime Architecture (simplified)

```
GitHub Actions (schedule or dispatch)
        │
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
  • connect Redis (with quota/OOM detection)
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
```

---

## 🔐 Security Notes

- Secrets live only in GitHub Secrets / environment; never in the repo.
- Container runs as non-root (`appuser`), read-only root filesystem, limited tmpfs.
- Redis and Telegram URLs/tokens are redacted in normal logging paths.
- Outcome data repo access is token-scoped.

---

## 📈 Monitoring Checklist

1. Watchdog Telegram alerts (silence or failed conclusion).
2. Workflow summary: duration, alerts sent, pairs scanned, memory, Redis status.
3. Artifacts: `bot-execution-logs-*` (7-day retention).
4. Redis: `KEYS pair_state:*`, `SCAN … MATCH recent_alert:*`, memory / command stats.
5. Outcome-data repo: daily JSONL growth and Brain reports under `reports/`.

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
