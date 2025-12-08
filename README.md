# 📡 Automated MACD Alert Bot  
**Dockerized Trading Alert System — Triggered Every 15 Minutes via Cron-jobs.org**

This repository contains a fully-containerized Python trading alert bot.  
It runs on GitHub and is triggered *externally* via **Cron-jobs.org**, which calls the GitHub API every 15 minutes to run the workflow and send alerts to Telegram.

The bot processes market data, evaluates MACD-based conditions, and sends formatted alerts to a Telegram chat.

---

## 🚀 Features

### 🔧 Technical
- Fully Dockerized using a production-friendly multi-stage build  
- GitHub Actions workflow to build & push images to GitHub Container Registry  
- Lightweight runtime layer with only required dependencies  
- Optimized startup speed (uv-based installation, bytecode compilation)

### 📊 Trading Logic
- Unified MACD strategy implemented in `src/macd_unified.py`
- Configurable via `config_macd.json`
- Designed to run every **15 minutes** on completed candles
- Supports multiple trading pairs (customizable)

### 📬 Notifications
- Sends alerts to Telegram via bot API  
- Includes pair name, MACD signal, confirmation, timestamps, etc.

### 🕒 Execution
- Triggered automatically using **Cron-jobs.org → GitHub API → Actions workflow**
- No reliance on GitHub's built-in cron

---

## 📁 Repository Structure

```
.
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── config_macd.json
├── wrapper.py
├── src/
│   └── macd_unified.py
└── .github/
    └── workflows/
        ├── build-and-push.yml
        └── run-bot.yml
```

---

## ⚙️ How It Works

### 1️⃣ GitHub Container Build  
The workflow **build-and-push.yml** automatically builds your Docker image and pushes it to GHCR.

### 2️⃣ Bot Execution (Triggered Externally)  
The **run-bot.yml** workflow runs *only when triggered* externally.  
Cron-jobs.org initiates a workflow dispatch every 15 minutes.

---

## 🔐 Required Secrets

| Secret Name | Purpose |
|------------|----------|
| `TELEGRAM_BOT_TOKEN` | Bot API token |
| `TELEGRAM_CHAT_ID` | Telegram destination |
| `API_KEY` | Market data API key |
| `GITHUB_TOKEN` | For workflow dispatch |
| `GHCR_PAT` | For private registry access |

---

## 🕒 Cron-jobs.org Setup

1. Create POST job:
```
https://api.github.com/repos/<user>/<repo>/actions/workflows/run-bot.yml/dispatches
```
2. Method: **POST**
3. Headers:
```
Authorization: token <GITHUB_TOKEN>
Accept: application/vnd.github+json
```
4. Body:
```json
{ "ref": "main" }
```
5. Schedule:
```
1,16,31,46 * * * *
```

---

## 🧠 Main Scripts

### `wrapper.py`
- Loads config  
- Fetches market data  
- Runs strategy  
- Sends Telegram alerts  

### `macd_unified.py`
- MACD calculation  
- Signal logic  
- Alert formatting  

---

## 🛠️ Configuration

Editable in `config_macd.json`:
```json
{
  "pairs": ["BTCUSDT", "ETHUSDT"],
  "fast": 12,
  "slow": 26,
  "signal": 9,
  "interval": "15m"
}
```

---

## 🐳 Local Development

Build:
```
docker build -t macd-bot .
```

Run:
```
docker run --rm -e TELEGRAM_BOT_TOKEN=xxx -e TELEGRAM_CHAT_ID=yyy -e API_KEY=zzz macd-bot
```

---

## 🛡️ Security

- Secrets stored only in GitHub Actions  
- No sensitive values inside repo  
- Minimal runtime image  

---

## 📞 Support

Need custom indicators? Multi-pair scanning? CI/CD improvements?  
Just ask!
