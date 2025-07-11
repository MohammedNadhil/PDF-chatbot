# chat.py
import openai
from dotenv import load_dotenv
from vector_db import VectorDB
from cache import QuestionCache
from preprocessing import PDFProcessor
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class PDFChat:
    def __init__(self, pdf_paths, source_id="default"):
        self.pdf_paths = pdf_paths
        self.source_id = source_id
        self.vector_db = VectorDB(source_id=source_id)
        self.cache = QuestionCache()

        if not self.vector_db.is_processed():
            self._initialize_vector_db()

    def _initialize_vector_db(self):
        processor = PDFProcessor(self.pdf_paths)
        chunks = processor.process_pdfs()
        self.vector_db.build_index(chunks)

    def ask(self, question):
        # Check cache first
        cached_answer = self.cache.get_answer(question)
        if cached_answer:
            return cached_answer
        
        # Get relevant context with score
        context, score = self.vector_db.get_relevant_context_with_score(question)
        SCORE_THRESHOLD = 0.3

        if context and score >= SCORE_THRESHOLD:
            system_prompt = (
                "You are a helpful assistant. Use the following context to answer the user's question. "
                "If the context doesn't provide enough information, say 'I don't know'."
            )
            messages = [
                {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
                {"role": "user", "content": question}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question briefly."},
                {"role": "user", "content": question}
            ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=messages
        )

        answer = response.choices[0].message.content.strip()
        self.cache.add_entry(question, answer)
        return answer

# --------------------------
# Usage Example
# --------------------------
if __name__ == "__main__":
    PDF_PATHS = ['doc1.pdf', 'doc2.pdf']
    SOURCE_ID = 'preloaded_docs'

    chat_system = PDFChat(PDF_PATHS, source_id=SOURCE_ID)
    
    while True:
        question = input("\nUser: ")
        if question.lower() in ['exit', 'quit']:
            break
        print("AI:", chat_system.ask(question))
