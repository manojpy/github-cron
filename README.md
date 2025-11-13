# Trading Alert Bots

This repository contains two automated trading alert bots that run every 15 minutes via [Cron-jobs.org](https://cron-jobs.org) and GitHub Actions.

### Bots
- **MACD Bot** – Detects PPO/MACD crossovers and sends Telegram alerts.
- **Fibonacci Pivot Bot** – Detects price interactions around Fibonacci pivot levels and alerts via Telegram.

### How It Works
1. Cron-jobs.org triggers GitHub Actions every 15 minutes.
2. Each bot fetches price data from Delta Exchange (`https://api.india.delta.exchange`).
3. Signals are analyzed and alerts are sent to Telegram.
4. State and log files are saved and pushed back to the repository.

### Secrets Used
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### Config Files
- `config_macd.json` – Settings for MACD Bot
- `config_fib.json` – Settings for Fibonacci Pivot Bot

Each bot writes to its own state and log file:
- `macd_state.sqlite` / `macd_bot.log`
- `fib_state.sqlite` / `fibonacci_pivot_bot.log`

---
