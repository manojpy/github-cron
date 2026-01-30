# 🤖 MACD Unified Bot

High-performance cryptocurrency trading alert bot with AOT compilation, Redis state management, and Telegram notifications. Runs on GitHub Actions every 15 minutes.

## 📋 Quick Overview

- **What**: Analyzes crypto pairs (BTCUSD, ETHUSD, etc) using 20+ technical indicators
- **When**: Runs on schedule via GitHub Actions cron (1, 16, 31, 46 minutes past every hour)
- **Outputs**: Sends trading alerts to Telegram with smart deduplication
- **Speed**: 25-35 seconds for 12 pairs (10-50x faster than pure Python via Numba AOT)
- **Memory**: <900MB footprint with aggressive garbage collection

## 🚀 Setup (5 Steps)

### 1. Fork & Configure Secrets
```bash
git clone https://github.com/manojpy/github-cron.git
cd github-cron
```

Add to GitHub **Settings → Secrets and variables → Actions**:
```
TELEGRAM_BOT_TOKEN     → Get from BotFather
TELEGRAM_CHAT_ID       → Your Telegram chat ID
REDIS_URL             → redis://user:pass@host:6379
DELTA_API_BASE        → https://api.india.delta.exchange
```

### 2. Edit Configuration
```bash
# Edit config_macd.json
nano config_macd.json
```

Key settings:
```json
{
  "PAIRS": ["BTCUSD", "ETHUSD", "AVAXUSD"],    // Trading pairs
  "PPO_FAST": 7, "PPO_SLOW": 16,              // Indicator periods
  "ENABLE_VWAP": true,                        // Features
  "ENABLE_PIVOT": true,
  "DRY_RUN_MODE": false                       // Test mode
}
```

### 3. Push & Build
```bash
git add config_macd.json
git commit -m "Configure bot"
git push
```

This triggers `build.yml` → builds Docker image with AOT compilation

### 4. Verify Build
- Check **Actions tab** → **Build AOT Image** 
- Wait for ✅ success (3-5 minutes)

### 5. Run Bot
- **Auto**: Bot runs on cron schedule
- **Manual**: Actions tab → **Run MACD Unified Bot** → **Run workflow**

Check results in Telegram inbox 📱

---

## ⚙️ Configuration Quick Reference

```json
{
  // REQUIRED (from GitHub Secrets)
  "TELEGRAM_BOT_TOKEN": "...",
  "TELEGRAM_CHAT_ID": "...",
  "REDIS_URL": "...",
  "DELTA_API_BASE": "https://api.india.delta.exchange",

  // Pairs to monitor
  "PAIRS": ["BTCUSD", "ETHUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD", "SOLUSD"],

  // Indicator periods
  "PPO_FAST": 7,           // PPO short EMA
  "PPO_SLOW": 16,          // PPO long EMA
  "PPO_SIGNAL": 5,         // PPO signal line
  "RMA_50_PERIOD": 50,     // 50-bar MA
  "RMA_200_PERIOD": 200,   // 200-bar MA
  "SRSI_RSI_LEN": 21,      // RSI period

  // Performance
  "MAX_PARALLEL_FETCH": 12,        // HTTP concurrency
  "RUN_TIMEOUT_SECONDS": 300,      // 5-minute max execution
  "HTTP_TIMEOUT": 8,               // Request timeout

  // Features
  "ENABLE_VWAP": true,             // Volume-weighted avg price
  "ENABLE_PIVOT": true,            // Support/resistance levels
  "CIRRUS_CLOUD_ENABLED": true,    // Trend indicator

  // Resilience
  "MEMORY_LIMIT_BYTES": 850000000, // 700MB soft limit
  "FAIL_ON_REDIS_DOWN": false,     // Degrade gracefully
  "FAIL_ON_TELEGRAM_DOWN": false   // Continue if Telegram fails
}
```

See [config_macd.json](config_macd.json) for all 40+ options.

---

## 📊 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.11 | Core bot logic |
| **Compilation** | Numba (JIT + AOT) | 20 indicator functions (30-50x speedup) |
| **Async** | asyncio | Concurrent API fetches, parallel evaluation |
| **State** | Redis | Alert deduplication, persistence |
| **Notifications** | Telegram Bot API | Alert delivery |
| **Deployment** | Docker + GitHub Actions | Automated build & execution |
| **Container** | Ubuntu 24.04 slim | 900MB memory limit |

---

## 📈 Indicators (20 Functions)

**Moving Averages**: EMA, RMA, SMA  
**Oscillators**: PPO, RSI, VWAP  
**Filters**: Kalman, Range Filter, Smooth Range  
**Momentum**: MMH (Magical Momentum Histogram)  
**Trends**: Cirrus Cloud (multi-scale filtering)  
**Patterns**: Wick quality checks, Pivot levels  
**Statistics**: Rolling std dev, min/max via monotonic deques

---

## 🔔 Alert Types (26 Signals)

| Category | Signals |
|----------|---------|
| **PPO** | Cross above/below signal, cross ±0, cross ±0.11 |
| **RSI** | Cross above/below 50 (with PPO guard) |
| **VWAP** | Cross above/below (20-min dedup) |
| **Pivots** | Cross above/below P, R1/R2/R3, S1/S2/S3 |
| **MMH** | Reversal UP, Reversal DOWN |

All alerts include: timestamp (IST), price, indicator values, wick quality.

---

## 🔧 Local Development

### Run Locally
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="src:$PYTHONPATH"
python src/macd_unified.py --validate-only  # Check config
python src/macd_unified.py --debug          # Run with debug logs
```

### Build AOT Binary
```bash
cd src
python aot_build.py --output-dir . --verify
cd ..
```

### Docker Test
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

### Redis connection failed
```
❌ Check REDIS_URL format: redis://user:pass@host:port
✅ Test: redis-cli -u "$REDIS_URL" ping
```

### Circuit breaker OPENED
```
❌ Delta API returning 5xx errors
✅ Wait 30s: bot auto-recovers
✅ Check: https://api.india.delta.exchange/status
```

### Memory limit exceeded
```
❌ Too many pairs (>15) or insufficient container memory
✅ Reduce PAIRS list or split into 2 bots
✅ Increase MEMORY_LIMIT_BYTES in Dockerfile
```

### Candle staleness error
```
❌ API data older than 20 minutes
✅ Increase MAX_CANDLE_STALENESS_SEC: 1800
```

### Rate limit exceeded
```
❌ Too many pairs or RATE_LIMIT_PER_MINUTE too high
✅ Lower RATE_LIMIT_PER_MINUTE: 60 → 45
✅ Reduce number of pairs
```

---

## 📁 Project Structure

```
github-cron/
├── src/
│   ├── macd_unified.py              (3.5K lines - main bot)
│   ├── numba_functions_shared.py    (1.2K lines - 20 JIT functions)
│   ├── aot_bridge.py                (250 lines - AOT/JIT fallback)
│   └── aot_build.py                 (400 lines - AOT compiler)
│
├── .github/workflows/
│   ├── build.yml                    (Docker build + AOT compile)
│   └── run-bot.yml                  (Execute bot on schedule)
│
├── config_macd.json                 (Configuration)
├── Dockerfile                       (Multi-stage: deps → AOT → runtime)
├── requirements.txt                 (Python dependencies)
├── .dockerignore
└── .gitignore

Total: ~5,350 lines Python
```

---

## 🎯 Architecture Overview

```
GitHub Actions (Cron every 15 min)
    ↓
build.yml (if code changed)
    ├─ Install deps (UV + pip)
    ├─ Compile AOT (aot_build.py → macd_aot_compiled.so)
    ├─ Build Docker image (multi-stage, 900MB)
    └─ Push to ghcr.io
    ↓
run-bot.yml (scheduled)
    ├─ Verify secrets (Telegram, Redis, Delta API)
    ├─ Pull Docker image
    ├─ Mount config_macd.json
    ├─ Run container (2 CPUs, 900MB memory, 5-min timeout)
    ├─ Fetch candles (parallel, 3 resolutions × 12 pairs)
    ├─ Calculate indicators (AOT compiled, ~5ms per pair)
    ├─ Evaluate alerts (check 26 conditions)
    ├─ Deduplicate (Redis Lua scripts)
    ├─ Send Telegram (batched, rate-limited)
    └─ Upload logs on failure
```

---

## 📊 Performance

| Task | AOT | JIT | Speedup |
|------|-----|-----|---------|
| Startup | 0.5s | 0.5s | 1x |
| PPO (350 bars) | 0.4ms | 12ms | **30x** |
| RSI (350 bars) | 0.3ms | 10ms | **33x** |
| 12 pairs, all indicators | 200ms | 2.5s | **12.5x** |
| Full cycle (fetch + eval + alert) | 25-35s | 30-40s | 1.2x |

---

## 🔐 Security

- ✅ Secrets never in repo (GitHub Secrets only)
- ✅ Redacted from logs (TOKEN, chat_id, redis:// masked)
- ✅ TLS 1.2+ for all API calls
- ✅ Non-root container user
- ✅ Read-only filesystem (except /tmp)
- ✅ OHLC validation on every candle
- ✅ Redis data TTL: 30 days auto-expiry

---

## 📈 Monitoring

### Check Logs
```bash
# GitHub Actions → Workflow run → Logs
# Or: Actions → Run MACD Unified Bot → View summary
```

### Manual Verification
```bash
# Validate config
python src/macd_unified.py --validate-only

# Check Redis state
redis-cli -u "$REDIS_URL" KEYS "pair_state:*" | head -5

# Check dedup window
redis-cli -u "$REDIS_URL" SCAN 0 MATCH "recent_alert:*"

# Watch Docker logs
docker logs -f macd_bot_runner
```

---

## 🤝 Support

**Issues**: Submit GitHub issues with logs + config (secrets redacted)  
**Questions**: Check Actions workflow summary for detailed report  
**Contributions**: PRs welcome for features, bug fixes, optimizations

---

## 📚 Resources

- [Numba Documentation](https://numba.readthedocs.io/)
- [Delta Exchange API](https://api.india.delta.exchange/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Redis Documentation](https://redis.io/docs/)

---

**Version**: 1.8.0-stable | **Last Updated**: 2025-01-22
