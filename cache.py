# 3. Cache System (cache.py)
# --------------------------
import json
from datetime import datetime, timedelta

class QuestionCache:
    def __init__(self, cache_file='qa_cache.json', expiry_days=7):
        self.cache_file = cache_file
        self.expiry_days = expiry_days
        self.cache = self._load_cache()

    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                return {k: v for k, v in data.items() 
                        if datetime.strptime(v['timestamp'], '%Y-%m-%d') > datetime.now() - timedelta(days=self.expiry_days)}
        except FileNotFoundError:
            return {}

    def get_answer(self, question):
        return self.cache.get(question, {}).get('answer')

    def add_entry(self, question, answer):
        self.cache[question] = {
            'answer': answer,
            'timestamp': datetime.now().strftime('%Y-%m-%d')
        }
        self._save_cache()

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)