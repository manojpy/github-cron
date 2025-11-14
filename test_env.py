# test_env.py
import os

def test_environment_variables():
    print("🔍 Checking Environment Variables...")
    print("=" * 50)
    
    # Check each required variable
    variables = {
        'NEWS_BOT_TOKEN': os.getenv('NEWS_BOT_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'), 
        'NEWSAPI_KEY': os.getenv('NEWSAPI_KEY')
    }
    
    all_set = True
    for var_name, var_value in variables.items():
        status = "✅ SET" if var_value else "❌ MISSING"
        print(f"{var_name}: {status}")
        if not var_value:
            all_set = False
    
    print("=" * 50)
    if all_set:
        print("🎉 All environment variables are set!")
        print("You can now run: python news_alerter.py")
    else:
        print("❌ Some environment variables are missing.")
        print("\n💡 Solution: Create a .env file with these variables:")
        print("NEWS_BOT_TOKEN=your_bot_token_here")
        print("TELEGRAM_CHAT_ID=your_chat_id_here") 
        print("NEWSAPI_KEY=your_newsapi_key_here")

if __name__ == "__main__":
    test_environment_variables()
