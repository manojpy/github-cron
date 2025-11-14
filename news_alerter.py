import os
import requests
import json
import hashlib
from datetime import datetime, timedelta

class GlobalNewsAlerter:
    def __init__(self):
        # Use NEW bot token, but same chat ID
        self.telegram_bot_token = os.getenv('8589118688:AAEYi9Uix03DiB7WQtU6Z7EJuQBlkFrSGEA')  # New bot token
        self.telegram_chat_id = os.getenv('203813932')  # Same chat ID as before!
        self.newsapi_key = os.getenv('6c5fd5cfb5d142bf917f038a3e1111eb')
        
        # ... rest of the code remains exactly the same ...
