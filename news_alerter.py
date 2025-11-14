import os
import requests
import json
import hashlib
from datetime import datetime, timedelta

class CryptoNewsAlerter:
    def __init__(self):
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        
        # Cryptocurrencies from your image
        self.crypto_keywords = {
            'BTC': ['bitcoin', 'btc', 'satoshi', 'bitcoin etf', 'bitcoin mining'],
            'ETH': ['ethereum', 'eth', 'vitalik', 'ethereum 2.0', 'merge', 'gas fee'],
            'BNB': ['binance', 'bnb', 'cz', 'changpeng zhao', 'binance coin'],
            'SOL': ['solana', 'sol', 'anatoly yakovenko', 'solana network'],
            'XRP': ['xrp', 'ripple', 'brad garlinghouse', 'sec vs ripple'],
            'ADA': ['cardano', 'ada', 'charles hoskinson', 'cardano network'],
            'DOT': ['polkadot', 'dot', 'gavin wood', 'parachain'],
            'AVAX': ['avalanche', 'avax', 'ava labs', 'avalanche network'],
            'LTC': ['litecoin', 'ltc', 'charlie lee', 'litecoin foundation'],
            'BCH': ['bitcoin cash', 'bch', 'bitcoin cash fork'],
            'SUI': ['sui', 'sui network', 'mysten labs'],
            'AAVE': ['aave', 'aave protocol', 'defi lending', 'flash loans']
        }
        
        # News categories that impact crypto markets
        self.impact_categories = {
            'regulation': [
                'regulation', 'sec', 'regulation', 'legal', 'law', 'government', 
                'tax', 'compliance', 'ban', 'restriction', 'framework', 'policy',
                'cfdc', 'finra', 'financial regulation', 'crypto bill', 'legislation'
            ],
            'adoption': [
                'adoption', 'institutional', 'etf approval', 'blackrock', 'fidelity',
                'vanguard', 'paypal', 'visa', 'mastercard', 'bank adoption',
                'corporate adoption', 'tesla', 'microstrategy', 'company investment'
            ],
            'technology': [
                'upgrade', 'hard fork', 'mainnet', 'launch', 'protocol', 'network',
                'blockchain', 'smart contract', 'scalability', 'transaction speed',
                'gas fees', 'congestion', 'outage', 'downtime', 'hack', 'exploit'
            ],
            'macro_economics': [
                'inflation', 'interest rate', 'federal reserve', 'fed', 'central bank',
                'economic policy', 'stimulus', 'quantitative easing', 'recession',
                'economic data', 'gdp', 'employment', 'jobs report', 'dollar'
            ],
            'security': [
                'hack', 'security breach', 'exploit', 'rug pull', 'scam', 'phishing',
                'wallet security', 'exchange hack', 'smart contract vulnerability',
                'flash crash', 'market manipulation', 'whale movement'
            ],
            'partnerships': [
                'partnership', 'collaboration', 'integration', 'alliance', 'deal',
                'enterprise', 'business development', 'strategic partnership'
            ]
        }
        
        # File to store sent news hashes
        self.sent_news_file = 'sent_news.json'
        
    def get_sent_news_hashes(self):
        """Load previously sent news hashes"""
        try:
            with open(self.sent_news_file, 'r') as f:
                return set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()
    
    def save_sent_news_hashes(self, hashes):
        """Save sent news hashes"""
        with open(self.sent_news_file, 'w') as f:
            json.dump(list(hashes), f)
    
    def create_news_hash(self, news_item):
        """Create unique hash for news item to avoid duplicates"""
        content = f"{news_item['title']}_{news_item['publishedAt']}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_crypto_impact_news(self):
        """Fetch news that could impact cryptocurrencies"""
        url = "https://newsapi.org/v2/everything"
        
        # Get news from last 3 hours for more comprehensive coverage
        from_date = (datetime.now() - timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%S')
        
        # Search for crypto and financial impact keywords
        search_keywords = [
            'cryptocurrency', 'bitcoin', 'ethereum', 'blockchain', 'digital currency',
            'crypto regulation', 'sec crypto', 'fed crypto', 'crypto market',
            'defi', 'nft', 'web3', 'token', 'digital asset'
        ]
        
        query = " OR ".join(search_keywords)
        
        params = {
            'apiKey': self.newsapi_key,
            'q': query,
            'language': 'en',
            'pageSize': 50,
            'sortBy': 'publishedAt',
            'from': from_date,
            'domains': 'bloomberg.com,reuters.com,coindesk.com,cointelegraph.com,decrypt.co,theblock.co'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            print(f"Found {len(articles)} crypto-related news articles")
            return articles
        except requests.exceptions.RequestException as e:
            print(f"Error fetching crypto news: {e}")
            return []
    
    def analyze_news_impact(self, news_items):
        """Analyze which cryptocurrencies are affected by the news and potential impact"""
        analyzed_news = []
        
        for item in news_items:
            title = item.get('title', '').lower()
            description = item.get('description', '').lower()
            content = f"{title} {description}"
            
            affected_cryptos = []
            impact_type = 'neutral'
            impact_category = 'general'
            
            # Check which cryptocurrencies are mentioned
            for crypto, keywords in self.crypto_keywords.items():
                if any(keyword in content for keyword in keywords):
                    affected_cryptos.append(crypto)
            
            # Determine impact category and type
            for category, keywords in self.impact_categories.items():
                if any(keyword in content for keyword in keywords):
                    impact_category = category
            
            # Determine positive/negative impact based on keywords
            positive_keywords = [
                'approval', 'adoption', 'partnership', 'integration', 'launch', 
                'upgrade', 'success', 'growth', 'bullish', 'rally', 'surge',
                'breakthrough', 'innovation', 'investment', 'funding', 'support'
            ]
            
            negative_keywords = [
                'ban', 'regulation', 'crackdown', 'lawsuit', 'sec', 'investigation',
                'hack', 'exploit', 'crash', 'collapse', 'scam', 'fraud', 'warning',
                'delay', 'problem', 'issue', 'outage', 'downtime', 'rejection'
            ]
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in content)
            negative_count = sum(1 for keyword in negative_keywords if keyword in content)
            
            if negative_count > positive_count:
                impact_type = 'negative'
            elif positive_count > negative_count:
                impact_type = 'positive'
            else:
                impact_type = 'neutral'
            
            if affected_cryptos:
                item['affected_cryptos'] = affected_cryptos
                item['impact_type'] = impact_type
                item['impact_category'] = impact_category
                analyzed_news.append(item)
        
        return analyzed_news
    
    def filter_new_news(self, news_items, sent_hashes):
        """Filter out already sent news"""
        new_news = []
        new_hashes = set()
        
        for item in news_items:
            news_hash = self.create_news_hash(item)
            if news_hash not in sent_hashes:
                new_news.append(item)
                new_hashes.add(news_hash)
        
        return new_news, new_hashes
    
    def format_impact_message(self, analyzed_news):
        """Format the impact analysis message for Telegram"""
        if not analyzed_news:
            return "📭 No significant crypto-impacting news in the past few hours."
        
        # Group news by impact type
        positive_news = [news for news in analyzed_news if news['impact_type'] == 'positive']
        negative_news = [news for news in analyzed_news if news['impact_type'] == 'negative']
        neutral_news = [news for news in analyzed_news if news['impact_type'] == 'neutral']
        
        message = "🚨 **Crypto Market Impact Alert** 🚨\n\n"
        message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        
        # Negative impacts first (most important for traders)
        if negative_news:
            message += "🔴 **Potential Negative Impacts** 🔴\n\n"
            for i, news in enumerate(negative_news[:5]):
                message += self.format_news_item(news, i + 1)
        
        # Positive impacts
        if positive_news:
            message += "🟢 **Potential Positive Impacts** 🟢\n\n"
            for i, news in enumerate(positive_news[:5]):
                message += self.format_news_item(news, i + 1)
        
        # Neutral/important developments
        if neutral_news and not (positive_news or negative_news):
            message += "⚪ **Market Developments** ⚪\n\n"
            for i, news in enumerate(neutral_news[:5]):
                message += self.format_news_item(news, i + 1)
        
        message += f"\n📊 **Summary**: {len(positive_news)} positive, {len(negative_news)} negative, {len(neutral_news)} neutral impacts"
        message += f"\n\n⏳ Next update in 1 hour"
        
        return message
    
    def format_news_item(self, news_item, index):
        """Format individual news item"""
        title = news_item.get('title', 'No title')
        source = news_item.get('source', {}).get('name', 'Unknown')
        url = news_item.get('url', '')
        affected_cryptos = news_item.get('affected_cryptos', [])
        impact_category = news_item.get('impact_category', 'general')
        
        # Shorten very long titles
        if len(title) > 100:
            title = title[:97] + "..."
        
        # Impact emojis
        impact_emojis = {
            'regulation': '⚖️',
            'adoption': '🏦',
            'technology': '💻',
            'macro_economics': '📊',
            'security': '🔒',
            'partnerships': '🤝',
            'general': '📰'
        }
        
        impact_emoji = impact_emojis.get(impact_category, '📰')
        
        formatted_item = f"{index}. {impact_emoji} **{title}**\n"
        
        if affected_cryptos:
            formatted_item += f"   💰 Affects: {', '.join(affected_cryptos)}\n"
        
        formatted_item += f"   📰 Source: {source}\n"
        
        if url:
            formatted_item += f"   🔗 [Read more]({url})\n"
        
        formatted_item += "\n"
        
        return formatted_item
    
    def send_telegram_message(self, message):
        """Send message to Telegram"""
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        
        payload = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            print("Message sent successfully to Telegram")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error sending Telegram message: {e}")
            return False
    
    def run(self):
        """Main function to run the crypto impact alerter"""
        print("Starting crypto impact news alerter...")
        
        # Check if required environment variables are set
        if not all([self.telegram_bot_token, self.telegram_chat_id, self.newsapi_key]):
            print("Error: Missing required environment variables")
            return False
        
        # Load previously sent news
        sent_hashes = self.get_sent_news_hashes()
        print(f"Loaded {len(sent_hashes)} previously sent news items")
        
        # Fetch crypto-impacting news
        print("Fetching crypto-impacting news...")
        news_items = self.get_crypto_impact_news()
        print(f"Found {len(news_items)} crypto-related news items")
        
        if not news_items:
            print("No crypto news items found")
            # Send a heartbeat message
            self.send_telegram_message(
                "✅ Crypto news check completed.\n"
                "No significant crypto-impacting news in the past few hours.\n\n"
                "Next update in 1 hour ⏰"
            )
            return True
        
        # Analyze impact
        analyzed_news = self.analyze_news_impact(news_items)
        print(f"Analyzed {len(analyzed_news)} news items with crypto impact")
        
        # Filter new news
        new_news, new_hashes = self.filter_new_news(analyzed_news, sent_hashes)
        print(f"Found {len(new_news)} new impactful news items")
        
        if not new_news:
            print("No new impactful news to send")
            self.send_telegram_message(
                "✅ Crypto news check completed.\n"
                "No new significant crypto-impacting news.\n\n"
                "Next update in 1 hour ⏰"
            )
            return True
        
        # Format and send message
        message = self.format_impact_message(new_news)
        success = self.send_telegram_message(message)
        
        if success:
            # Update sent news hashes
            updated_hashes = sent_hashes.union(new_hashes)
            # Keep only last 500 hashes to prevent file from growing too large
            if len(updated_hashes) > 500:
                updated_hashes = set(list(updated_hashes)[-500:])
            self.save_sent_news_hashes(updated_hashes)
            print(f"Updated sent news hashes: {len(updated_hashes)} items")
        
        return success

def main():
    alerter = CryptoNewsAlerter()
    success = alerter.run()
    
    if success:
        print("Crypto impact news alerter completed successfully")
    else:
        print("Crypto impact news alerter failed")
        exit(1)

if __name__ == "__main__":
    main()
