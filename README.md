📊 MACD Unified Alert Bot

A high-performance, asynchronous market scanner for Delta Exchange. This bot uses Numba AOT (Ahead-of-Time) compilation to achieve ultra-fast technical analysis, delivering real-time alerts via Telegram.

🕒 Execution Schedule
The bot is triggered externally via Cron-jobs.org to ensure precise timing:Interval: Every 15 minutes.Timing: 1, 16, 31, 46 minutes past the hour.Purpose: This 1-minute offset allows the exchange to finalize the 15m candle data before the bot starts scanning.🛠 Project StructureThe repository is organized to separate high-performance math from automation logic:
MACD-Unified/
├── .github/workflows/
│   ├── build.yml             # CI: Compiles AOT binaries & builds Docker image
│   └── run-bot.yml           # CD: Executes the bot scan on trigger
├── src/
│   ├── macd_unified.py       # Main Entry Point: Orchestrates the scan
│   ├── numba_functions_shared.py # Math Core: TA indicators and logic
│   ├── aot_build.py          # Compiler: Generates native .so libraries
│   └── aot_bridge.py         # Loader: Dispatches between AOT and JIT
├── config_macd.json          # Configuration: Pairs, PPO periods, etc.
├── requirements.txt          # Dependencies: Numba, Redis, Aiohttp
├── Dockerfile                # Multi-stage optimized build
├── .gitignore                # Git exclusion rules
└── .dockerignore             # Docker build exclusion rules

🚀 Key Salient Points

1. High-Performance Math (AOT)
Unlike standard Python bots, this project uses aot_build.py to compile math functions into a native Linux shared library (.so).Benefit: Zero "cold-start" latency. The bot runs at full speed from the very first second.Fallback: If the AOT library is missing, aot_bridge.py automatically falls back to standard JIT compilation.
  
2. Intelligent AlertingDeduplication:
Uses Redis to store the state of sent alerts. You won't get spammed with the same signal multiple times.Async Engine: Scans all pairs (BTC, ETH, SOL, etc.) simultaneously using non-blocking I/O.Rate Limiting: Built-in Telegram throttling to prevent the bot from being banned by Telegram during high volatility.

3.Automated CI/CDBuild Pipeline: 
Whenever you change the code in src/, GitHub Actions automatically recompiles the math and updates your Docker image.Runtime Safety: The run-bot.yml workflow mounts your config_macd.json at runtime, allowing you to update pairs or settings without needing a full code rebuild.

⚙️ Setup & Configuration
Prerequisites
Redis: Required for alert persistence.
GitHub Secrets: 
The following must be set in your repo settings:
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
REDIS_URL

Local Execution
To run the bot manually:Install requirements: pip install -r requirements.txt
Run the scanner: python src/macd_unified.py

📊 Indicator Logic
The bot calculates a "Unified" signal using:
PPO (Percentage Price Oscillator): Faster and more accurate than standard Macd
MMH for cross-asset comparison.
Kalman Filters: To smooth out price noise.
Cirrus Clouds: A custom trend-following indicator.
Wick Detection: Filters out "fake" breakouts by analyzing candle wicks.
