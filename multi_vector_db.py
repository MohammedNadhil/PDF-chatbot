# multi_vector_db.py
import os
from typing import List, Tuple, Dict, Optional
from vector_db import VectorDB
from sentence_transformers import SentenceTransformer

class MultiVectorDB:
    """
    Handles multiple vector databases for cross-document search
    """
    def __init__(self, source_ids: List[str]):
        self.source_ids = source_ids
        self.vector_dbs = {}
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load all vector databases
        for source_id in source_ids:
            try:
                self.vector_dbs[source_id] = VectorDB(source_id=source_id)
                print(f"Loaded vector database for source: {source_id}")
            except Exception as e:
                print(f"Failed to load source {source_id}: {e}")
    
    def get_relevant_context_from_all_sources(self, query: str, top_k: int = 3) -> Tuple[str, float, List[dict]]:
        """
        Search across all sources and return combined context with metadata
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return per source
            
        Returns:
            Tuple[str, float, List[dict]]: (combined_context, best_score, source_metadata)
        """
        all_results = []
        
        for source_id, vector_db in self.vector_dbs.items():
            if not vector_db.is_processed():
                print(f"Source {source_id} not processed, skipping...")
                continue
                
            try:
                context, score = vector_db.get_relevant_context_with_score(query, top_k)
                if context and score > 0:
                    all_results.append({
                        "source_id": source_id,
                        "context": context,
                        "score": float(score),  # Convert numpy.float32 to Python float
                        "context_length": len(context)
                    })
                    print(f"Found relevant context in {source_id} with score {score:.3f}")
            except Exception as e:
                print(f"Error searching in source {source_id}: {e}")
        
        if not all_results:
            return "", 0.0, []
        
        # Sort by score and get top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:top_k]
        
        # Combine contexts from top sources
        combined_context = "\n\n---\n\n".join([r["context"] for r in top_results])
        best_score = float(top_results[0]["score"])  # Convert to Python float
        
        return combined_context, best_score, top_results
    
    def is_any_processed(self) -> bool:
        """
        Check if any of the sources have been processed
        
        Returns:
            bool: True if at least one source is processed
        """
        return any(vector_db.is_processed() for vector_db in self.vector_dbs.values())
    
    def get_processed_sources(self) -> List[str]:
        """
        Get list of processed source IDs
        
        Returns:
            List[str]: List of processed source IDs
        """
        return [source_id for source_id, vector_db in self.vector_dbs.items() 
                if vector_db.is_processed()]
    
    def get_source_stats(self) -> Dict[str, dict]:
        """
        Get statistics for all sources
        
        Returns:
            Dict[str, dict]: Statistics for each source
        """
        stats = {}
        for source_id, vector_db in self.vector_dbs.items():
            stats[source_id] = {
                "processed": vector_db.is_processed(),
                "text_count": len(vector_db.texts) if hasattr(vector_db, 'texts') else 0
            }
        return stats
