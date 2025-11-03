# import os
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from typing import List

# class VectorDB:
#     def __init__(self, index_path='vector_index.faiss', texts_path='processed_texts.json'):
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.index_path = index_path
#         self.texts_path = texts_path
#         self.index = None
#         self.texts = []
        
#         # Load existing index if available
#         self.load_index()

#     def build_index(self, new_chunks: List[str]):
#         """
#         Build or append to existing vector index
        
#         Args:
#             new_chunks (List[str]): New text chunks to be indexed
#         """
#         # If no existing index, create a new one
#         if self.index is None:
#             self.texts = new_chunks
#             embeddings = self.model.encode(new_chunks, show_progress_bar=True)
#             self.index = faiss.IndexFlatL2(embeddings.shape[1])
#             self.index.add(np.array(embeddings).astype('float32'))
#         else:
#             # Append new chunks to existing index
#             existing_embedding_count = len(self.texts)
            
#             # Combine existing and new texts
#             self.texts.extend(new_chunks)
            
#             # Encode new chunks
#             new_embeddings = self.model.encode(new_chunks, show_progress_bar=True)
            
#             # Add new embeddings to the index
#             self.index.add(np.array(new_embeddings).astype('float32'))
        
#         # Save updated index and texts
#         self.save_index()
#         self._save_texts()
   

#     def save_index(self):
#         """Save the FAISS index to file"""
#         if self.index is not None:
#             faiss.write_index(self.index, self.index_path)

#     def load_index(self):
#         """Load existing FAISS index if available"""
#         try:
#             if os.path.exists(self.index_path):
#                 self.index = faiss.read_index(self.index_path)
#                 # Load associated texts
#                 self._load_texts()
#                 return True
#         except Exception as e:
#             print(f"Error loading index: {e}")
#         return False

#     def _save_texts(self):
#         """Save processed texts to a JSON file"""
#         import json
#         with open(self.texts_path, 'w') as f:
#             json.dump(self.texts, f)

#     def _load_texts(self):
#         """Load processed texts from JSON file"""
#         import json
#         try:
#             with open(self.texts_path, 'r') as f:
#                 self.texts = json.load(f)
#         except FileNotFoundError:
#             self.texts = []

#     def get_relevant_context(self, query: str, top_k: int = 3) -> str:
#         """
#         Perform semantic search to find most relevant text chunks
        
#         Args:
#             query (str): Input query to search against
#             top_k (int): Number of top relevant chunks to return
        
#         Returns:
#             str: Concatenated most relevant text chunks
#         """
#         if self.index is None or len(self.texts) == 0:
#             return "No documents have been processed yet."
        
#         # Encode the query
#         query_embedding = self.model.encode([query]).astype('float32')
        
#         # Perform similarity search
#         distances, indices = self.index.search(query_embedding, top_k)
        
#         # Retrieve the most relevant text chunks
#         relevant_chunks = [self.texts[idx] for idx in indices[0]]
        
#         return "\n\n".join(relevant_chunks)

#     def is_processed(self) -> bool:
#         """
#         Check if documents have been processed
        
#         Returns:
#             bool: True if documents are processed, False otherwise
#         """
#         return self.index is not None and len(self.texts) > 0

#vector_db.py
import os,json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from preprocessing import PDFProcessor
from typing import Tuple


class VectorDB:
    def __init__(self, source_id: str = "preloaded"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.source_id = source_id

        os.makedirs("vector_indices", exist_ok=True)
        os.makedirs("processed_texts", exist_ok=True)

        self.index_path = f'vector_indices/{source_id}.faiss'
        self.texts_path = f'processed_texts/{source_id}.json'

        self.index = None
        self.texts = []

        self.load_index()
    
    def preload_pdfs(self, pdf_directory='preloaded_pdfs'):
        """
        Load PDFs from a specified directory and build the vector index
        
        Args:
            pdf_directory (str): Directory containing preloaded PDFs
        
        Returns:
            bool: True if PDFs were loaded successfully, False otherwise
        """
        try:
            # Check if directory exists
            if not os.path.exists(pdf_directory):
                os.makedirs(pdf_directory)
                print(f"Created directory {pdf_directory}, but no PDFs found.")
                return False
            
            # Get all PDF files in the directory
            pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) 
                        if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                print(f"No PDF files found in {pdf_directory}")
                return False
            
            # Process PDFs
            processor = PDFProcessor(pdf_files)
            chunks = processor.process_pdfs()
            
            # Build vector index
            self.build_index(chunks)
            
            print(f"Successfully loaded {len(pdf_files)} PDFs with {len(chunks)} chunks.")
            return True
            
        except Exception as e:
            print(f"Error preloading PDFs: {e}")
            return False

    def build_index(self, new_chunks: List[str]):
        """
        Build or append to existing vector index
        
        Args:
            new_chunks (List[str]): New text chunks to be indexed
        """
        # If no existing index, create a new one
        if self.index is None:
            self.texts = new_chunks
            embeddings = self.model.encode(new_chunks, show_progress_bar=True)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings).astype('float32'))
        else:
            # Append new chunks to existing index
            existing_embedding_count = len(self.texts)
            
            # Combine existing and new texts
            self.texts.extend(new_chunks)
            
            # Encode new chunks
            new_embeddings = self.model.encode(new_chunks, show_progress_bar=True)
            
            # Add new embeddings to the index
            self.index.add(np.array(new_embeddings).astype('float32'))
        
        # Save updated index and texts
        self.save_index()
        self._save_texts()

    def save_index(self):
        """Save the FAISS index to file"""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def load_index(self):
        """Load existing FAISS index if available"""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                self._load_texts()
                return True
        except Exception as e:
            print(f"Error loading index: {e}")
        return False
    
    def _save_texts(self):
        """Save processed texts to a JSON file"""
        with open(self.texts_path, 'w') as f:
            json.dump(self.texts, f)

    def _load_texts(self):
        """Load processed texts from JSON file"""
        try:
            with open(self.texts_path, 'r') as f:
                self.texts = json.load(f)
        except FileNotFoundError:
            self.texts = []

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """
        Perform semantic search to find most relevant text chunks
        
        Args:
            query (str): Input query to search against
            top_k (int): Number of top relevant chunks to return
        
        Returns:
            str: Concatenated most relevant text chunks
        """
        if self.index is None or len(self.texts) == 0:
            return "No documents have been processed yet."
        
        # Encode the query
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Perform similarity search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the most relevant text chunks
        relevant_chunks = [self.texts[idx] for idx in indices[0]]
        
        return "\n\n".join(relevant_chunks)

    def is_processed(self) -> bool:
        """
        Check if documents have been processed
        
        Returns:
            bool: True if documents are processed, False otherwise
        """
        return self.index is not None and len(self.texts) > 0

    def get_relevant_context_with_score(self, query: str, top_k: int = 3) -> Tuple[str, float]:
        """
        Perform semantic search and return most relevant context along with similarity score.
    
        Args:
            query (str): The input query
            top_k (int): Number of top chunks to consider (default 3)
    
        Returns:
            tuple: (Concatenated relevant text chunks, top similarity score [0.0 - 1.0])
        """
        if self.index is None or len(self.texts) == 0:
            return "", 0.0

        # Encode the query
        query_embedding = self.model.encode([query]).astype('float32')

        # Perform similarity search
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve the most relevant text chunks
        relevant_chunks = [self.texts[idx] for idx in indices[0]]

        # FAISS IndexFlatL2 returns squared L2 distances (lower is better)
        # To convert distance to a pseudo-similarity score (higher is better), use:
        top_distance = distances[0][0]
        similarity_score = float(1 / (1 + top_distance))  # Normalize score between 0 and 1, convert to Python float

        context = "\n\n".join(relevant_chunks)

        return context, similarity_score
