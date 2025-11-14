import os
import requests
import json
import hashlib
from datetime import datetime, timedelta

# ⚠️ REPLACE THESE WITH YOUR ACTUAL VALUES ⚠️
NEWS_BOT_TOKEN = "8589118688:AAEYi9Uix03DiB7WQtU6Z7EJuQBlkFrSGEA"  # ← Your NEW bot token from @BotFather
TELEGRAM_CHAT_ID = "203813932"                           # ← Your personal Chat ID (same as before)
NEWSAPI_KEY = "6c5fd5cfb5d142bf917f038a3e1111eb"                 # ← Your NewsAPI key from newsapi.org

class GlobalNewsAlerter:
    def __init__(self):
        # Use the direct values
        self.telegram_bot_token = NEWS_BOT_TOKEN
        self.telegram_chat_id = TELEGRAM_CHAT_ID
        self.newsapi_key = NEWSAPI_KEY
        
        # Debug info
        print(f"🔧 Debug Info:")
        print(f"   Token: {'✅' if self.telegram_bot_token else '❌'}")
        print(f"   Chat ID: {'✅' if self.telegram_chat_id else '❌'}")
        print(f"   NewsAPI: {'✅' if self.newsapi_key else '❌'}")
        
        if not all([self.telegram_bot_token, self.telegram_chat_id, self.newsapi_key]):
            print("❌ ERROR: Please replace the token values at the top of this file!")
            return
        
        # Major impactful news categories
        self.categories = {
            'politics': [
                'election', 'government', 'president', 'prime minister', 'parliament',
                'congress', 'senate', 'diplomacy', 'summit', 'treaty', 'sanctions',
                'international relations', 'united nations', 'nato', 'alliance'
            ],
            'business': [
                'economy', 'recession', 'inflation', 'gdp', 'employment', 'jobs',
                'trade', 'tariffs', 'market', 'corporate', 'merger', 'acquisition',
                'bankruptcy', 'layoffs', 'economic crisis', 'financial'
            ],
            'stock_market': [
                'stock market', 'dow jones', 's&p 500', 'nasdaq', 'market crash',
                'market rally', 'trading', 'investor', 'wall street', 'bull market',
                'bear market', 'market volatility', 'financial markets'
            ],
            'sports': [
                'world cup', 'olympics', 'championship', 'tournament', 'final',
                'victory', 'defeat', 'record', 'champion', 'super bowl', 'premier league',
                'nba finals', 'world series', 'grand slam'
            ],
            'technology': [
                'breakthrough', 'innovation', 'ai', 'artificial intelligence',
                'quantum computing', 'cyber attack', 'data breach', 'hack',
                'spacex', 'nasa', 'mars', 'moon mission', 'scientific discovery'
            ],
            'disasters': [
                'earthquake', 'tsunami', 'hurricane', 'typhoon', 'flood', 'wildfire',
                'tornado', 'volcano', 'natural disaster', 'emergency', 'evacuation',
                'rescue', 'death toll', 'catastrophe'
            ],
            'wars': [
                'war', 'conflict', 'military', 'attack', 'invasion', 'defense',
                'nuclear', 'missile', 'drone', 'casualties', 'ceasefire', 'peace talks',
                'terrorism', 'terror attack', 'military operation'
            ],
            'health': [
                'pandemic', 'epidemic', 'outbreak', 'virus', 'health emergency',
                'who', 'cdc', 'vaccine', 'medical breakthrough', 'hospital',
                'health crisis', 'public health'
            ]
        }
        
        self.high_impact_keywords = [
            'crisis', 'emergency', 'breakthrough', 'historic', 'record', 'unprecedented',
            'major', 'significant', 'important', 'critical', 'urgent', 'alert',
            'warning', 'catastrophe', 'disaster', 'tragedy', 'victory', 'landmark',
            'milestone', 'revolution', 'breakthrough', 'collapse', 'surge', 'plunge',
            'soar', 'crash', 'attack', 'strike', 'summit', 'deal', 'agreement'
        ]
        
        self.sent_news_file = 'sent_news.json'
    
    def get_sent_news_hashes(self):
        try:
            with open(self.sent_news_file, 'r') as f:
                return set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()
    
    def save_sent_news_hashes(self, hashes):
        with open(self.sent_news_file, 'w') as f:
            json.dump(list(hashes), f)
    
    def create_news_hash(self, news_item):
        content = f"{news_item['title']}_{news_item['publishedAt']}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_global_news(self):
        sources = [
            'reuters', 'associated-press', 'bbc-news', 'cnn', 
            'al-jazeera-english', 'the-guardian-uk', 'the-new-york-times'
        ]
        
        url = "https://newsapi.org/v2/top-headlines"
        from_date = (datetime.now() - timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%S')
        
        all_articles = []
        
        for source in sources:
            params = {
                'apiKey': self.newsapi_key,
                'sources': source,
                'pageSize': 20,
                'sortBy': 'publishedAt',
                'from': from_date
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    all_articles.extend(articles)
                    print(f"✅ Found {len(articles)} from {source}")
            except Exception as e:
                print(f"❌ Error from {source}: {e}")
                continue
        
        # Also get general headlines
        params = {
            'apiKey': self.newsapi_key,
            'language': 'en',
            'pageSize': 30,
            'sortBy': 'publishedAt',
            'from': from_date
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                all_articles.extend(articles)
                print(f"✅ Found {len(articles)} general headlines")
        except Exception as e:
            print(f"❌ Error getting general headlines: {e}")
        
        # Remove duplicates
        seen_urls = set()
        unique_articles = []
        
        for article in all_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(url)
        
        print(f"📊 Total unique articles: {len(unique_articles)}")
        return unique_articles
    
    def filter_high_impact_news(self, news_items):
        high_impact_news = []
        
        for item in news_items:
            title = item.get('title', '').lower()
            description = item.get('description', '').lower()
            content = f"{title} {description}"
            
            impact_score = sum(1 for keyword in self.high_impact_keywords if keyword in content)
            
            category_match = False
            for category, keywords in self.categories.items():
                if any(keyword in content for keyword in keywords):
                    category_match = True
                    break
            
            if impact_score >= 2 or category_match:
                positive_keywords = [
                    'breakthrough', 'victory', 'success', 'achievement', 'record',
                    'milestone', 'landmark', 'peace', 'agreement', 'deal', 'surge',
                    'rally', 'growth', 'recovery', 'innovation', 'discovery'
                ]
                
                negative_keywords = [
                    'crisis', 'emergency', 'disaster', 'tragedy', 'attack', 'war',
                    'conflict', 'crash', 'collapse', 'death', 'killed', 'injured',
                    'outbreak', 'pandemic', 'recession', 'layoffs', 'bankruptcy'
                ]
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in content)
                negative_count = sum(1 for keyword in negative_keywords if keyword in content)
                
                if negative_count > positive_count:
                    impact_type = 'negative'
                elif positive_count > negative_count:
                    impact_type = 'positive'
                else:
                    impact_type = 'significant'
                
                item['impact_type'] = impact_type
                item['impact_score'] = impact_score
                high_impact_news.append(item)
        
        high_impact_news.sort(key=lambda x: x['impact_score'], reverse=True)
        print(f"🎯 High-impact stories: {len(high_impact_news)}")
        return high_impact_news
    
    def categorize_news(self, news_items):
        categorized = {category: [] for category in self.categories.keys()}
        categorized['other'] = []
        
        for item in news_items:
            title = item.get('title', '').lower()
            description = item.get('description', '').lower()
            content = f"{title} {description}"
            
            categorized_flag = False
            for category, keywords in self.categories.items():
                if any(keyword in content for keyword in keywords):
                    categorized[category].append(item)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append(item)
        
        return categorized
    
    def filter_new_news(self, news_items, sent_hashes):
        new_news = []
        new_hashes = set()
        
        for item in news_items:
            news_hash = self.create_news_hash(item)
            if news_hash not in sent_hashes:
                new_news.append(item)
                new_hashes.add(news_hash)
        
        return new_news, new_hashes
    
    def format_news_message(self, categorized_news):
        if not any(categorized_news.values()):
            return "📭 No major impactful news in the past few hours."
        
        message = "🌍 **Global Impact News Alert** 🌍\n\n"
        message += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        
        emojis = {
            'politics': '🏛️',
            'business': '💼',
            'stock_market': '📈',
            'sports': '⚽',
            'technology': '💻',
            'disasters': '⚠️',
            'wars': '⚔️',
            'health': '🏥',
            'other': '📰'
        }
        
        impact_emojis = {
            'positive': '🟢',
            'negative': '🔴',
            'significant': '🟡'
        }
        
        total_stories = sum(len(items) for items in categorized_news.values())
        message += f"📊 **Today's Major Stories**: {total_stories} impactful events\n\n"
        
        for category, items in categorized_news.items():
            if items:
                emoji = emojis.get(category, '📰')
                message += f"{emoji} **{category.replace('_', ' ').title()}**\n"
                
                for i, item in enumerate(items[:3]):
                    impact_emoji = impact_emojis.get(item.get('impact_type', 'significant'), '🟡')
                    title = item.get('title', 'No title')
                    source = item.get('source', {}).get('name', 'Unknown')
                    url = item.get('url', '')
                    
                    if len(title) > 90:
                        title = title[:87] + "..."
                    
                    message += f"{impact_emoji} {title}\n"
                    message += f"   📰 {source}\n"
                    if url:
                        message += f"   🔗 [Read more]({url})\n"
                    message += "\n"
                
                message += "\n"
        
        impact_counts = {}
        for category, items in categorized_news.items():
            for item in items:
                impact_type = item.get('impact_type', 'significant')
                impact_counts[impact_type] = impact_counts.get(impact_type, 0) + 1
        
        if impact_counts:
            summary_parts = []
            if 'negative' in impact_counts:
                summary_parts.append(f"🔴 {impact_counts['negative']} critical")
            if 'positive' in impact_counts:
                summary_parts.append(f"🟢 {impact_counts['positive']} positive")
            if 'significant' in impact_counts:
                summary_parts.append(f"🟡 {impact_counts['significant']} significant")
            
            message += f"**Impact Summary**: {', '.join(summary_parts)}\n"
        
        message += f"\n⏰ Next update in 1 hour"
        
        return message
    
    def send_telegram_message(self, message):
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        
        payload = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        try:
            print("📤 Sending message to Telegram...")
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            print("✅ Message sent successfully to Telegram")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Error sending Telegram message: {e}")
            return False
    
    def run(self):
        print("🚀 Starting global impact news alerter...")
        
        # Check if tokens are set
        if not all([self.telegram_bot_token, self.telegram_chat_id, self.newsapi_key]):
            print("❌ ERROR: Please replace the token values at the top of this file!")
            print("   Make sure you've updated:")
            print("   - NEWS_BOT_TOKEN (your new bot token)")
            print("   - TELEGRAM_CHAT_ID (your personal chat ID)") 
            print("   - NEWSAPI_KEY (from newsapi.org)")
            return False
        
        # Load previously sent news
        sent_hashes = self.get_sent_news_hashes()
        print(f"📁 Loaded {len(sent_hashes)} previously sent news items")
        
        # Fetch global news
        print("📡 Fetching global news...")
        news_items = self.get_global_news()
        
        if not news_items:
            print("❌ No news items found")
            self.send_telegram_message(
                "✅ Global news check completed.\n"
                "No major impactful news in the past few hours.\n\n"
                "Next update in 1 hour ⏰"
            )
            return True
        
        # Filter for high-impact news
        high_impact_news = self.filter_high_impact_news(news_items)
        
        if not high_impact_news:
            print("❌ No high-impact news found")
            self.send_telegram_message(
                "✅ Global news check completed.\n"
                "No major impactful news in the past few hours.\n\n"
                "Next update in 1 hour ⏰"
            )
            return True
        
        # Filter new news
        new_news, new_hashes = self.filter_new_news(high_impact_news, sent_hashes)
        print(f"🆕 New high-impact stories: {len(new_news)}")
        
        if not new_news:
            print("ℹ️ No new high-impact news to send")
            self.send_telegram_message(
                "✅ Global news check completed.\n"
                "No new major impactful news.\n\n"
                "Next update in 1 hour ⏰"
            )
            return True
        
        # Categorize and send
        categorized_news = self.categorize_news(new_news)
        message = self.format_news_message(categorized_news)
        success = self.send_telegram_message(message)
        
        if success:
            updated_hashes = sent_hashes.union(new_hashes)
            if len(updated_hashes) > 1000:
                updated_hashes = set(list(updated_hashes)[-1000:])
            self.save_sent_news_hashes(updated_hashes)
            print(f"💾 Updated sent news hashes: {len(updated_hashes)} items")
        
        return success

def main():
    alerter = GlobalNewsAlerter()
    success = alerter.run()
    
    if success:
        print("🎉 Global impact news alerter completed successfully")
    else:
        print("💥 Global impact news alerter failed")
        exit(1)

if __name__ == "__main__":
    main()
