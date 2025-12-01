# Unified Trading Alert Bot

Automated cryptocurrency trading signal detection system using technical indicators.

## 🚀 Quick Start

### Prerequisites
- GitLab account with Container Registry enabled
- Redis instance (managed or self-hosted)
- Telegram bot token
- Cron-jobs.org account (or similar scheduler)

### Setup

1. **Clone the repository**
```bash
   git clone https://gitlab.com/yourusername/trading-bot.git
   cd trading-bot
```

2. **Set GitLab CI/CD Variables**
   
   Go to `Settings → CI/CD → Variables` and add:
   
   | Variable | Value | Type |
   |----------|-------|------|
   | `TELEGRAM_BOT_TOKEN` | Your bot token | Masked |
   | `TELEGRAM_CHAT_ID` | Your chat ID | Masked |
   | `REDIS_URL` | `redis://user:pass@host:port` | Masked |

3. **Build the Docker image**
```bash
   # Commit and push to trigger initial build
   git add .
   git commit -m "Initial setup"
   git push origin main
```

4. **Configure Cron Trigger**
   
   URL: `https://gitlab.com/api/v4/projects/YOUR_PROJECT_ID/trigger/pipeline?token=YOUR_TRIGGER_TOKEN&ref=main`
   
   Schedule: `1,16,31,46 * * * *` (every 15 minutes at :01, :16, :31, :46)

## 📊 Monitored Pairs

- BTC/USD, ETH/USD, AVAX/USD, BCH/USD
- XRP/USD, BNB/USD, LTC/USD, DOT/USD
- ADA/USD, SUI/USD, AAVE/USD, SOL/USD

## 🔧 Configuration

Edit `config_macd.json` to customize:
- Indicator parameters (PPO, RSI, RMA)
- Alert thresholds
- Batch processing settings
- Timeout values

## 📈 Technical Indicators

- **PPO (Percentage Price Oscillator)**: Momentum indicator
- **Smooth RSI**: Kalman-filtered RSI
- **Cirrus Cloud**: Custom trend filter
- **VWAP**: Volume-weighted average price
- **Pivot Points**: Support/resistance levels
- **MMH (Magical Momentum Histogram)**: Custom reversal detector

## 🛠️ Local Development
```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Set environment variables
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export REDIS_URL="redis://localhost:6379"

# Run once
python gitlab_wrapper.py
```

## 🐳 Docker Build & Run
```bash
# Build image locally
docker build -t trading-bot:local .

# Run container
docker run --rm \
  -e TELEGRAM_BOT_TOKEN="your_token" \
  -e TELEGRAM_CHAT_ID="your_chat_id" \
  -e REDIS_URL="redis://host:6379" \
  trading-bot:local
```

## 📝 Logs

Logs are available in GitLab CI artifacts:
- `Job artifacts → last_run.log`

## 🔒 Security Notes

- Never commit secrets to repository
- Use GitLab CI/CD variables for sensitive data
- Container runs as non-root user
- SSL/TLS enabled for all external connections

## 📄 License

[Your License Here]

## 🤝 Contributing

[Your contribution guidelines]
