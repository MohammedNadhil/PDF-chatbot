#semantic_cache.py
import json
from datetime import datetime, timedelta
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional

class SemanticQuestionCache:
    def __init__(self, cache_file='semantic_qa_cache.json', expiry_days=7, similarity_threshold=0.85):
        self.cache_file = cache_file
        self.expiry_days = expiry_days
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Same model as vector DB for consistency
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                
                # Filter out expired entries
                current_date = datetime.now()
                filtered_data = {}
                
                for key, entry in data.items():
                    entry_date = datetime.strptime(entry['timestamp'], '%Y-%m-%d')
                    if entry_date > current_date - timedelta(days=self.expiry_days):
                        filtered_data[key] = entry
                
                return filtered_data
            return {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    
    def _compute_embedding(self, question: str) -> np.ndarray:
        """Compute embedding for a question"""
        return self.model.encode([question])[0]
    
    def _find_similar_question(self, question: str) -> Tuple[Optional[str], float]:
        """Find the most similar question in the cache"""
        if not self.cache:
            return None, 0.0
        
        # Compute embedding for the current question
        query_embedding = self._compute_embedding(question)
        
        best_match = None
        highest_similarity = 0.0
        
        # Compare with cached questions
        for cached_key, entry in self.cache.items():
            cached_embedding = np.array(entry['embedding'])
            
            # Compute cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = cached_key
        
        return best_match, highest_similarity
    
    def get_answer(self, question: str) -> Optional[str]:
        """Get an answer for a semantically similar question if it exists in the cache"""
        # First try exact match
        if question in self.cache:
            print(f"Exact cache hit for: {question}")
            return self.cache[question]['answer']
        
        # Then try semantic match
        similar_question, similarity = self._find_similar_question(question)
        
        if similar_question and similarity >= self.similarity_threshold:
            print(f"Semantic cache hit ({similarity:.2f}) for: {question}")
            print(f"Similar question: {similar_question}")
            return self.cache[similar_question]['answer']
        
        return None
    
    def add_entry(self, question: str, answer: str) -> None:
        """Add a new question-answer pair to the cache"""
        embedding = self._compute_embedding(question)
        
        self.cache[question] = {
            'answer': answer,
            'embedding': embedding.tolist(),  # Convert numpy array to list for JSON serialization
            'timestamp': datetime.now().strftime('%Y-%m-%d')
        }
        
        self._save_cache()
    
    def _save_cache(self) -> None:
        """Save the cache to a file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4) 
        except Exception as e:
            print(f"Error saving cache: {e}")