# 📡 🤖 MACD Unified Alert Bot
A high-performance, asynchronous cryptocurrency market scanner designed for Delta Exchange. This bot utilizes Numba AOT (Ahead-of-Time) compilation to achieve sub-second technical analysis across multiple trading pairs, delivering real-time alerts via Telegram.
🚀 Key Features
Hybrid Execution: Combines Python's flexibility with C-like performance via Numba AOT compilation.
Asynchronous Engine: Built on aiohttp and asyncio for concurrent data fetching and processing.
Persistent State: Uses Redis for tracking alert states and avoiding duplicate notifications across runs.
Advanced TA: Implements MMH, PPO, RSI, Kalman Filters, VWAP, and custom "Cirrus Cloud" indicators.
Production Ready: Multi-stage Docker builds, resource-constrained execution, and automated GitHub Actions workflows.
🛠 Project Structure
The project follows a modular structure to separate performance-critical math from application logic.
MACD-Unified/
├── .github/
│   └── workflows/
│       ├── build.yml          # CI: Builds & pushes the AOT Docker image
│       └── run-bot.yml        # CD: Executes the bot (triggered by cron)
├── src/
│   ├── macd_unified.py        # Main entry point & Application logic
│   ├── numba_functions_shared.py # Math & TA logic (Source of Truth)
│   ├── aot_build.py           # Compiler script to generate .so libraries
│   └── aot_bridge.py          # Runtime dispatcher (AOT vs JIT fallback)
├── config_macd.json           # User configuration & parameters
├── requirements.txt           # Python dependencies (Numba, Redis, Pydantic)
├── Dockerfile                 # High-optimization multi-stage build
├── .gitignore  & .dockerignore       # Excludes caches, .so files, and envs
└── README.md                  # Project documentation



⚙️ Technical Architecture
1. Performance: AOT vs JIT
Most Python trading bots suffer from "Cold Start" latency because Numba compiles functions the first time they are called.
AOT (Ahead-of-Time): We pre-compile functions into a native .so library during the Docker build stage.
Result: Zero latency on the first calculation. The bot is "warm" from second one.
2. Execution Flow
The bot is designed to be stateless yet persistent:
Trigger: Cron-jobs.org pings a GitHub Webhook or Workflow dispatch.
Fetch: Parallel GET requests to Delta Exchange for OHLCV data.
Compute: Vectorized math via the pre-compiled AOT module.
Deduplicate: Query Redis to see if the signal was already sent.
Alert: Ship formatted messages to Telegram.
Exit: Cleanly close connections to minimize billing seconds.
📋 Configuration
The config_macd.json file controls the bot's behavior. Sensitive credentials should be injected via GitHub Secrets.
Key
Description
PAIRS
List of Delta Exchange symbols (e.g., ["BTCUSD", "ETHUSD"])
PPO_FAST/SLOW
Parameters for the Percentage Price Oscillator
REDIS_URL
Connection string for alert state persistence
RUN_TIMEOUT_SECONDS
Safety cutoff for the execution window

🤖 Automation & Deployment
Cron Scheduling
The bot is triggered externally via Cron-jobs.org at the following intervals (UTC):
1, 16, 31, 46 minutes past the hour.
This ensures data for the previous 15-minute candle is fully closed and processed.
CI/CD Pipeline
Build Workflow (build.yml):
Triggers on changes to src/ or Dockerfile.
Compiles math functions into native binaries.
Pushes a hardened image to GitHub Packages (GHCR).
Run Workflow (run-bot.yml):
Pulls the latest AOT-optimized image.
Mounts config_macd.json.
Executes the scan and provides a summary in the GitHub Step Summary.
🛠 Local Development
To run or compile the bot locally:
Install Dependencies:
pip install -r requirements.txt



Compile AOT (Optional):
python src/aot_build.py --output-dir ./src



Run the Bot:
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_id"
python src/macd_unified.py



🛡 Salient Points
Memory Management: The bot enforces an 850MB limit (configurable) to prevent OOM errors in serverless environments.
Sanitization: All incoming exchange data is passed through a Numba-optimized sanitization loop to handle NaN or Inf values without crashing.
Graceful Shutdown: Implements signal handling for SIGTERM to ensure Redis connections are closed cleanly.

