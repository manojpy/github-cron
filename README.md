# Trading Alert Bots

This repository contains an automated trading alert bot that run every 15 minutes via [Cron-jobs.org](https://cron-jobs.org) and GitHub Actions.

### Bots
- **MACD Bot** – Detects PPO/MACD/Vwap/Pivots crossovers and sends Telegram alerts.
### How It Works
1. Cron-jobs.org triggers GitHub Actions every 15 minutes.
2. The bot fetches price data from Delta Exchange (`https://api.india.delta.exchange`).
3. Signals are analyzed and alerts are sent to Telegram.
4. State and log files are saved and pushed back to Upstash Redis.

### Secrets Used
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `REDIS_URL`

### Config Files
- `config_macd.json` – Settings for MACD
Bot writes the state and log file to Upstash Redis
---
