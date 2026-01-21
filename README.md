# 🤖 MACD Unified Bot - Comprehensive Guide

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [API Integration](#api-integration)
- [Technical Details](#technical-details)
- [Deployment](#deployment)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)
- [Project Structure](#project-structure)

---

## 🎯 Overview

**MACD Unified Bot** is a high-performance, production-grade cryptocurrency trading alert system that combines **Ahead-of-Time (AOT) compilation** with **JIT fallback capabilities** to deliver rapid technical analysis and real-time Telegram notifications.

### What It Does
- Fetches OHLCV candle data for multiple cryptocurrency pairs from Delta Exchange API
- Calculates 20+ advanced technical indicators in real-time (PPO, RSI, VWAP, Pivot Levels, MMH, etc.)
- Detects trading signals across multiple timeframes (5m, 15m, daily)
- Sends aggregated alerts to Telegram with smart deduplication
- Runs via serverless GitHub Actions on a 15-minute schedule (or custom intervals)

### Why It's Special
- **Hybrid Compilation**: AOT-compiled Numba functions for 10-50x speedup vs pure Python
- **Memory Efficient**: <900MB footprint with aggressive garbage collection
- **Resilient**: Redis-backed state persistence, circuit breakers, rate limiting
- **Observable**: Structured logging with correlation IDs, trace context
- **Scalable**: Parallel async fetching for 12+ pairs in <35 seconds

---

## 🏗️ Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions Trigger                    │
│              (Cron: 1, 16, 31, 46 minutes)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  build.yml - Build & Push Image   │
        │  (AOT Compilation + Docker Build) │
        └───────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  run-bot.yml - Execute Bot        │
        │  (Pull Config, Mount Secrets)     │
        └─────────────┬─────────────────────┘
                      │
                      ▼
        ┌───────────────────────────────────┐
        │   Docker Container (ubuntu-24.04) │
        │  - Python 3.11 + Numba            │
        │  - AOT Binary (macd_aot_compiled) │
        │  - JIT Fallback Ready             │
        └─────────────┬─────────────────────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
   HTTP Fetch    Redis State     Telegram API
   Delta API     Management      Notifications
       │              │              │
       └──────────────┼──────────────┘
                      │
                      ▼
        ┌───────────────────────────────────┐
        │  macd_unified.py - Main Bot       │
        │  ├─ Fetch candles (parallel)      │
        │  ├─ Calculate indicators (AOT)    │
        │  ├─ Evaluate alerts               │
        │  ├─ Deduplicate (Redis)           │
        │  └─ Send Telegram (batched)       │
        └───────────────────────────────────┘
```

### Component Breakdown

| Component | File | Purpose |
|-----------|------|---------|
| **Main Entry Point** | `macd_unified.py` | Orchestrates entire bot logic, async coordination |
| **Numba Functions** | `numba_functions_shared.py` | 20 JIT-compiled indicator calculations |
| **AOT Bridge** | `aot_bridge.py` | Transparent fallback between AOT binary & JIT |
| **AOT Compiler** | `aot_build.py` | Compiles shared functions to `.so` binary |
| **Config Schema** | `config_macd.json` | Settings (pairs, periods, limits, etc.) |
| **Build Pipeline** | `.github/workflows/build.yml` | Docker image compilation & push |
| **Execution Pipeline** | `.github/workflows/run-bot.yml` | Bot execution with secret management |
| **Container** | `Dockerfile` | Multi-stage: deps, AOT, runtime |

---

## ✨ Key Features

### 1. **Technical Indicators (20 Total)**

#### Moving Averages & Trends
- **PPO** (Percentage Price Oscillator): 7/16/5 periods
- **EMA**: Exponential Moving Average with warm-up
- **RMA**: Wilder's Moving Average (EMA variant)
- **SMA**: Simple Moving Average via rolling window

#### Filters & Smoothing
- **Kalman Filter**: Dynamic trend estimation (R=0.01, Q=0.1)
- **Range Filter**: Deviation-based smoothing
- **Cirrus Cloud**: Multi-scale trend (X1=22, X2=9, X3=15, X4=5)

#### Oscillators & Momentum
- **RSI**: Relative Strength Index (21-period)
- **VWAP**: Volume-Weighted Average Price (daily reset)
- **MMH**: Magical Momentum Histogram (custom log-odds transform)
- **Smooth RSI**: RSI + Kalman filter combo

#### Volatility & Support/Resistance
- **Rolling Std Dev**: Standard deviation with responsiveness factor
- **Pivot Levels**: P, R1/R2/R3, S1/S2/S3 (15-bar lookback)

#### Pattern Recognition
- **Wick Quality Check**: Buy/sell candle validation via wick ratio
- **Rolling Min/Max**: Monotonic deque-based extrema tracking

### 2. **Alert System (26+ Signal Types)**

#### PPO-Based Signals
- ✅ PPO cross above/below signal line
- ✅ PPO cross above/below zero
- ✅ PPO cross above/below ±0.11 thresholds

#### RSI-Based Signals
- ✅ RSI cross above/below 50 (with PPO guard)

#### VWAP Signals
- ✅ Price cross above/below VWAP

#### Pivot Signals
- ✅ Price cross above/below each pivot level (P, R1/R2/R3, S1/S2/S3)
- ✅ Distance-based filtering (±1.5% from actual cross)

#### Momentum Signals
- ✅ MMH Reversal BUY (uptrend + positive reversal)
- ✅ MMH Reversal SELL (downtrend + negative reversal)

### 3. **Smart Deduplication**
- **Temporal Windows**: 10-minute rolling window per alert
- **Redis Lua Scripts**: Atomic dedup checks
- **Batch Operations**: Pipelined state management
- **Fallback**: In-memory dedup if Redis degraded

### 4. **Resilience Mechanisms**

| Mechanism | Purpose | Config |
|-----------|---------|--------|
| **Circuit Breaker** | Fail-fast on API cascades | Threshold=3, Timeout=60s |
| **Rate Limiter** | Respect API quotas | 60/min to Delta, 25/min Telegram |
| **Token Bucket** | Smooth Telegram dispatch | Burst=8 messages |
| **Retry Logic** | Exponential backoff | Base=0.8s, Max=30s |
| **Redis Lock** | Prevent concurrent runs | 900s expiry, auto-extend |
| **Memory Limits** | Prevent OOM crashes | 700MB soft, 900MB hard |

### 5. **Data Validation**
- **OHLC Integrity**: Verify L≤O,H≤C,H≥L for all candles
- **Timestamp Monotonicity**: Ensure chronological ordering
- **NaN/Inf Detection**: Replace invalid values safely
- **Candle Staleness**: Max 20-minute old data (1200s)
- **Price Sanity**: No >50% jumps between candles

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)
- GitHub account with Actions enabled
- Redis instance (cloud or self-hosted)
- Telegram Bot Token (from BotFather)
- Delta Exchange API access

### Local Development Setup

#### 1. Clone Repository
```bash
git clone https://github.com/manojpy/github-cron.git
cd github-cron
```

#### 2. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Application
```bash
# Copy example config
cp config_macd.json config_macd_local.json

# Edit with your credentials
nano config_macd_local.json
```

#### 5. Build AOT Binary (Optional)
```bash
cd src
python aot_build.py --output-dir . --module-name macd_aot_compiled --verify
cd ..
```

#### 6. Run Bot Locally
```bash
export PYTHONPATH="src:$PYTHONPATH"
python src/macd_unified.py --validate-only  # Validate config first
python src/macd_unified.py --debug          # Run with debug logging
```

### Docker Local Testing
```bash
# Build image locally
docker build -t macd-unified-local:latest .

# Run with mounted config
docker run --rm \
  -e TELEGRAM_BOT_TOKEN="your_token" \
  -e TELEGRAM_CHAT_ID="your_chat_id" \
  -e REDIS_URL="redis://..." \
  -e DELTA_API_BASE="https://api.india.delta.exchange" \
  -v $(pwd)/config_macd.json:/app/src/config_macd.json:ro \
  macd-unified-local:latest
```

---

## ⚙️ Configuration

### config_macd.json Structure

#### Required Secrets (from GitHub Secrets / Environment)
```json
{
  "TELEGRAM_BOT_TOKEN": "123456:ABC-DEF_ghijk...",
  "TELEGRAM_CHAT_ID": "-1001234567890",
  "REDIS_URL": "redis://user:password@host:6379/0",
  "DELTA_API_BASE": "https://api.india.delta.exchange"
}
```

#### Trading Pairs
```json
{
  "PAIRS": [
    "BTCUSD", "ETHUSD", "AVAXUSD", "BCHUSD", "XRPUSD",
    "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD",
    "AAVEUSD", "SOLUSD"
  ]
}
```

#### Indicator Periods
```json
{
  "PPO_FAST": 7,           // Fast EMA period
  "PPO_SLOW": 16,          // Slow EMA period
  "PPO_SIGNAL": 5,         // Signal line EMA
  "RMA_50_PERIOD": 50,     // 50-bar MA
  "RMA_200_PERIOD": 200,   // 200-bar MA
  "SRSI_RSI_LEN": 21,      // RSI period
  "SRSI_KALMAN_LEN": 5,    // Kalman filter smoothing
  "SRSI_EMA_LEN": 5        // Final EMA smoothing
}
```

#### Cirrus Cloud Parameters
```json
{
  "CIRRUS_CLOUD_ENABLED": true,
  "X1": 22,  // Primary range filter period
  "X2": 9,   // Primary range mult (2*X1-1)
  "X3": 15,  // Secondary filter period
  "X4": 5    // Secondary range mult
}
```

#### Performance Tuning
```json
{
  "MAX_PARALLEL_FETCH": 12,        // HTTP concurrency
  "HTTP_TIMEOUT": 8,               // Request timeout (s)
  "CANDLE_FETCH_RETRIES": 2,       // Retry attempts
  "CANDLE_FETCH_BACKOFF": 1.2,     // Exponential base
  "RUN_TIMEOUT_SECONDS": 300,      // Total execution limit (5 min)
  "BATCH_SIZE": 4,                 // Evaluation batch size
  "TCP_CONN_LIMIT": 16,            // Max HTTP connections
  "TCP_CONN_LIMIT_PER_HOST": 12    // Per-host limit
}
```

#### Resilience & Limits
```json
{
  "MEMORY_LIMIT_BYTES": 700000000,       // Soft: 700 MB
  "STATE_EXPIRY_DAYS": 30,               // Redis key TTL
  "FAIL_ON_REDIS_DOWN": false,           // Degraded mode allowed
  "FAIL_ON_TELEGRAM_DOWN": false,        // Continue if Telegram fails
  "TELEGRAM_RATE_LIMIT_PER_MINUTE": 25,  // Safe from throttling
  "TELEGRAM_BURST_SIZE": 8,              // Max concurrent sends
  "REDIS_CONNECTION_RETRIES": 2,         // Retry attempts
  "REDIS_RETRY_DELAY": 1.5,              // Delay between retries
  "MAX_ALERTS_PER_PAIR": 8,              // Suppress spam
  "CB_FAILURE_THRESHOLD": 5,             // Circuit breaker opens at 5 failures
  "CB_RECOVERY_TIMEOUT": 30              // Retry after 30s
}
```

#### Feature Flags
```json
{
  "DEBUG_MODE": false,               // Enable verbose logging
  "SEND_TEST_MESSAGE": false,        // Send startup notification
  "ENABLE_VWAP": true,              // Volume-weighted avg price
  "ENABLE_PIVOT": true,             // Support/resistance levels
  "PIVOT_LOOKBACK_PERIOD": 15,      // Daily bars for pivots
  "NUMBA_PARALLEL": true,           // Parallel JIT compilation
  "SKIP_WARMUP": true,              // Skip Numba JIT warmup
  "DRY_RUN_MODE": false             // Log alerts, don't send
}
```

#### API & Cache
```json
{
  "PRODUCTS_CACHE_TTL": 43200,           // 12 hours
  "MAX_CANDLE_STALENESS_SEC": 1200,      // 20 minutes max
  "RATE_LIMIT_PER_MINUTE": 60,          // Requests/min
  "ALERT_DEDUP_WINDOW_SEC": 600,        // 10-min dedup window
  "CANDLE_PUBLICATION_LAG_SEC": 45      // Expected API delay
}
```

### Configuration Validation

The bot validates all settings on startup:

```bash
# Check config syntax
python src/macd_unified.py --validate-only

# Output should be:
# âœ… Configuration validated successfully | Pairs: 12 | Workers: 12 | Timeout: 300s
```

---

## 🌐 API Integration

### Delta Exchange API

#### Endpoint: Chart History
```
GET https://api.india.delta.exchange/v2/chart/history
```

**Parameters:**
| Param | Example | Purpose |
|-------|---------|---------|
| `symbol` | `BTCUSD` | Trading pair |
| `resolution` | `15` / `5` / `D` | Timeframe (minutes or 'D' for daily) |
| `from` | `1706000000` | Start timestamp (unix seconds) |
| `to` | `1706100000` | End timestamp (unix seconds) |

**Response:**
```json
{
  "result": {
    "t": [1706000000, 1706000900, ...],  // timestamps
    "o": [45000.50, 45010.25, ...],      // opens
    "h": [45050.00, 45020.75, ...],      // highs
    "l": [44990.00, 44995.50, ...],      // lows
    "c": [45020.00, 45015.00, ...],      // closes
    "v": [123.45, 234.56, ...]           // volumes
  }
}
```

#### Rate Limits
- **Current**: 60 requests/minute
- **Bot Usage**: ~15-18 requests per run (4-5 per pair × 3-4 resolutions)
- **Overhead**: Retries + circuit breaker probes = ~10% extra

### Telegram Bot API

#### Endpoint: Send Message
```
POST https://api.telegram.org/bot{TOKEN}/sendMessage
```

**Parameters:**
| Param | Type | Example |
|-------|------|---------|
| `chat_id` | int | `-1001234567890` |
| `text` | string | "🟢 BTC USD alert..." |
| `parse_mode` | enum | `MarkdownV2` |

**Rate Limits:**
- **Hard Limit**: 30 messages/minute per bot
- **Bot Config**: 25 messages/minute (safe margin)
- **Burst**: 8 concurrent requests (token bucket)

#### Alert Message Format

**Single Alert:**
```
🟢 BTCUSD - $45,123.45
PPO cross above signal | Wick 15.2% | MMH (0.34)
22-01-2025 14:30 IST
```

**Batch Alert (3+ signals):**
```
🟢 **ETHUSD** • $2,450.00  22-01-2025 14:45 IST
├─ 🟢 PPO cross above signal | PPO 0.15 vs Sig 0.12 | Wick 18% | MMH (0.45)
├─ 🟢 PPO cross above 0 | PPO 0.02 | Wick 18% | MMH (0.45)
└─ 📈 Price cross above VWAP | VWAP $2445.50 | Wick 18% | MMH (0.45)
```

---

## 🔧 Technical Details

### Numba JIT Compilation

#### Why Numba?
- **Speed**: 10-50x faster than pure NumPy for loops
- **Memory**: Direct array operations, no Python overhead
- **Simplicity**: Write Python, compile to machine code

#### The 20 Compiled Functions

```python
# Sanitization (2 functions)
sanitize_array_numba()              # Replace NaN/Inf with default
sanitize_array_numba_parallel()     # Same, but parallel

# Moving Averages (2 functions)
ema_loop()                          # EMA with period α
ema_loop_alpha()                    # EMA with explicit alpha (for RMA)

# Filters (3 functions)
kalman_loop()                       # Kalman filter for trend
rng_filter_loop()                   # Range filter smoothing
smooth_range()                      # Two-stage EMA smoothing

# Extrema & Statistics (3 functions)
rolling_min_max_numba()             # Min/max via monotonic deques
rolling_std()                       # Rolling std dev with responsiveness
rolling_mean_numba()                # Simple moving average

# Oscillators (2 functions)
calculate_ppo_core()                # PPO + signal line
calculate_rsi_core()                # RSI momentum oscillator

# MMH Components (3 functions)
calc_mmh_worm_loop()                # Deviation-limited close tracking
calc_mmh_value_loop()               # Normalized -1..+1 value
calc_mmh_momentum_loop()            # Log-odds momentum transform
calc_mmh_momentum_smoothing()       # Final recursion smoothing

# Market Data (1 function)
vwap_daily_loop()                   # VWAP with daily reset

# Trends (1 function)
calculate_trends_with_state()       # Cloud up/down determination

# Patterns (2 functions)
vectorized_wick_check_buy()         # Green candle + wick quality
vectorized_wick_check_sell()        # Red candle + wick quality
```

### AOT vs JIT Trade-offs

| Factor | AOT | JIT |
|--------|-----|-----|
| **Startup** | Instant | 2-10s warmup |
| **Speed (first run)** | 100% | 10-30% (no compilation) |
| **Speed (subsequent)** | Baseline | 90-95% of AOT |
| **Size** | +50MB binary | -50MB, pure .py |
| **Debugging** | Harder | Easier (Python stack) |
| **Fallback** | None (or slow) | Always works |
| **Build Complexity** | High | Zero |

### Data Flow: Single Pair Evaluation

```
PAIR: BTCUSD
│
├─ FETCH: [15m, 5m, daily] candles in parallel (3 HTTP calls)
│  ├─ 15m: 350 candles (87.5 hours @ 15m)
│  ├─ 5m: 350 candles (29.2 hours @ 5m)
│  └─ daily: 15 candles (15 days)
│
├─ PARSE: Convert JSON → NumPy arrays
│  └─ Validate OHLC, remove NaNs, check timestamps
│
├─ CALCULATE: Compute all indicators via Numba
│  ├─ PPO (7/16/5): ~500µs
│  ├─ RSI (21): ~400µs
│  ├─ VWAP: ~600µs
│  ├─ Pivots: ~50µs
│  ├─ Cirrus Cloud: ~700µs
│  ├─ MMH (144): ~2000µs
│  └─ Total: ~4-5ms
│
├─ EVALUATE: Check 26 alert conditions
│  ├─ Get previous alert states from Redis
│  ├─ Evaluate each condition (0-60 triggers total)
│  ├─ Atomically update Redis state
│  └─ Filter duplicates via dedup window
│
└─ ALERT: Send triggered alerts to Telegram
   └─ Batch if >1 alert, single if =1

Total Time Per Pair: ~100-300ms
Total Time All 12 Pairs: 1200-3600ms (parallel)
```

### Memory Efficiency

**Per-Run Memory Profile:**

```
Startup: ~50 MB
└─ Python interpreter, numpy, redis, aiohttp

Fetch Phase: +150 MB
└─ 12 pairs × 3 resolutions × 350 candles × 8 bytes

Indicator Phase: +100 MB
└─ Intermediate arrays (PPO, RSI, VWAP, etc.)

Evaluation Phase: -250 MB
└─ Cleanup: release fetch data, release indicators
└─ GC triggers at 85% of memory limit

Cleanup Phase: ~50 MB
└─ Final state: logs + redis connection only

Peak Usage: ~280-350 MB
Limit: 700 MB (safe margin 2x)
Hard Limit: 900 MB (OOM killer protection)
```

### Asynchronous Architecture

The bot uses `asyncio` for concurrent operations:

```
Main Event Loop
├─ run_once() - orchestration
│  ├─ Establish Redis connection
│  ├─ Acquire distributed lock
│  ├─ Launch lock extension task
│  │  └─ Every 5min: extend lock TTL (900s → 900s)
│  │
│  └─ process_pairs_with_workers()
│     ├─ Phase 1: Fetch all candles in parallel
│     │  ├─ fetch_candles() × 36 (12 pairs × 3 resolutions)
│     │  └─ Rate limiter: 12 concurrent, 60 req/min
│     │
│     ├─ Phase 2: Prepare evaluation tasks
│     │  └─ Parse JSON to NumPy (sync)
│     │
│     └─ Phase 3: Evaluate pairs in parallel
│        ├─ guarded_eval() × 12 pairs
│        ├─ Async to sync via asyncio.to_thread()
│        └─ Numba functions (nogil, no GIL contention)
│
└─ Cleanup & teardown
   ├─ Release Redis lock
   ├─ Close Redis connection pool
   ├─ Close HTTP session
   └─ Final GC

Total Concurrency: ~48 coroutines at peak
Semaphores: Limit to 12 concurrent HTTP + evaluations
```

---

## 📦 Deployment

### GitHub Actions Setup

#### 1. Configure Repository Secrets

Go to **Settings → Secrets and variables → Actions** and add:

```
TELEGRAM_BOT_TOKEN     = "123456:ABC-DEF..."
TELEGRAM_CHAT_ID       = "-1001234567890"
REDIS_URL             = "redis://user:pass@host:6379"
DELTA_API_BASE        = "https://api.india.delta.exchange"
```

#### 2. Configure Cron Schedule

Edit `.github/workflows/run-bot.yml`:

```yaml
on:
  schedule:
    # Run at 1, 16, 31, 46 minutes (every 15 minutes, 1-minute delay)
    - cron: '1,16,31,46 * * * *'
    # Timezone: UTC (adjust for your region)
```

**For India (IST = UTC+5:30), adjust:
```yaml
# To run at :00, :15, :30, :45 IST, use:
cron: '30,45 18,19,20,21,22,23 * * *'  # Evening IST
```

#### 3. Trigger Builds

The workflow automatically:
1. **On `build.yml`**: Build Docker image when code changes
2. **On `run-bot.yml`**: Execute bot on schedule

Manual trigger via Actions tab:
```
Actions > Run MACD Unified Bot > Run workflow > Dry run mode
```

### Deployment Architecture

```
GitHub Actions (Ephemeral Runner)
│
├─ build.yml
│  ├─ Checkout code
│  ├─ Setup buildx (Docker)
│  ├─ Login to GHCR
│  └─ Build multi-stage Dockerfile
│     ├─ Stage 1: UV installer
│     ├─ Stage 2: Deps builder (numba, numpy, etc)
│     ├─ Stage 3: AOT compiler (aot_build.py)
│     └─ Stage 4: Runtime image (900MB)
│  └─ Push to ghcr.io/manojpy/github-cron/macd-unified-aot
│
└─ run-bot.yml (triggered by schedule)
   ├─ Checkout config_macd.json
   ├─ Verify config syntax
   ├─ Check GitHub Secrets present
   ├─ Pull image from GHCR
   └─ Run container
      ├─ Mount config from repo
      ├─ Inject secrets as env vars
      ├─ Limit to 2 CPUs, 900 MB memory
      ├─ Set 330s timeout (5m bot + 30s cleanup)
      └─ Parse results and report
```

### Container Security

- **User**: Non-root `appuser` (UID 1000)
- **Filesystem**: Read-only root, writable `/tmp` only
- **Network**: `--network host` (for GitHub runner)
- **Capabilities**: No new privileges
- **Health**: Disabled (save CPU for bot)

---

## 📊 Monitoring & Troubleshooting

### Logging Levels

```bash
# Default: INFO
python src/macd_unified.py

# Verbose: DEBUG
python src/macd_unified.py --debug

# Only validate config
python src/macd_unified.py --validate-only
```

### Log Format

```
2025-01-22 14:45:30.123 | INFO     | macd_bot | [abc12def] | run_once:4521 | 🎯 Run started | Correlation ID: abc12def | Reference time: 1706000730 (22-01-2025 14:45:30 IST)
```

**Fields:**
- **Timestamp**: `YYYY-MM-DD HH:MM:SS.mmm`
- **Level**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Logger**: Module name
- **Trace ID**: Correlation ID for this run
- **Location**: `function:line_number`
- **Message**: Structured log message

### Common Issues & Solutions

#### Issue: "Redis connection failed"
```
Symptom: âŒ Redis connection failed after all retries
Cause: REDIS_URL invalid or Redis instance down
Solution:
  1. Test connection: redis-cli -u "$REDIS_URL" ping
  2. Verify URL format: redis://user:pass@host:port/db
  3. Check firewall: telnet host port
  4. Set FAIL_ON_REDIS_DOWN=false for degraded mode
```

#### Issue: "Circuit breaker OPEN"
```
Symptom: Circuit breaker: OPENED after 5 failures
Cause: Delta API returning 5xx errors
Solution:
  1. Check API status: https://api.india.delta.exchange/status
  2. Wait 30s: bot auto-recovers (CB_RECOVERY_TIMEOUT)
  3. Increase timeout: CB_RECOVERY_TIMEOUT: 60
```

#### Issue: "Memory limit exceeded"
```
Symptom: ðŸš¨ Memory limit exceeded at startup (750MB / 700MB)
Cause: Too many pairs or not enough free container memory
Solution:
  1. Reduce PAIRS list: limit to <15 pairs
  2. Lower constants: MIN_CANDLES_FOR_INDICATORS to 200
  3. Increase memory: Edit Dockerfile MEMORY_LIMIT_BYTES
  4. Run separately: Split into 2 bots with different pairs
```

#### Issue: "Candle staleness"
```
Symptom: Selected candle is stale: 1500s old (max: 1200s)
Cause: API publishing delayed beyond expected window
Solution:
  1. Increase tolerance: MAX_CANDLE_STALENESS_SEC: 1800
  2. Check trigger time: Is bot running >45s past :00?
  3. Verify API: Fetch manually and check timestamps
```

#### Issue: "Rate limit exceeded"
```
Symptom: Rate limit reached (25/25), sleeping 60.2s
Cause: Too many pairs or RATE_LIMIT_PER_MINUTE too high
Solution:
  1. Lower RATE_LIMIT_PER_MINUTE: 45 → 40
  2. Reduce pairs or timeframes
  3. Increase RUN_TIMEOUT_SECONDS if safe
```

### Monitoring Dashboard (Manual)

Create an observability setup:

```bash
# Watch logs in real
