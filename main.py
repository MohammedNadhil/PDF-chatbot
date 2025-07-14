from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import PDFProcessor
from vector_db import VectorDB
from semantic_cache import SemanticQuestionCache
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

import os
import hashlib
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not found in environment variables")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="TTII Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache and preloaded index
cache = SemanticQuestionCache()
preloaded_db = VectorDB(source_id="preloaded")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting TTII Chatbot API...")
    try:
        # Don't block startup with PDF processing
        import asyncio
        asyncio.create_task(preload_pdfs_async())
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

async def preload_pdfs_async():
    try:
        logger.info("Starting PDF preloading...")
        VectorDB(source_id="preloaded").preload_pdfs("preloaded_pdfs")
        logger.info("PDF preloading completed")
    except Exception as e:
        logger.error(f"PDF preloading failed: {e}")

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "TTII Chatbot API is running"}

@app.get("/health")
async def detailed_health_check():
    try:
        # Check if OpenAI API key is available
        api_key_status = "available" if os.getenv("OPENAI_API_KEY") else "missing"
        
        return {
            "status": "healthy",
            "message": "TTII Chatbot API is running",
            "openai_api_key": api_key_status,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class QuestionRequest(BaseModel):
    question: str
    source_id: str | None = None  # Optional

@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    try:
        for file in files:
            contents = await file.read()
            source_id = hashlib.md5(contents).hexdigest()

            folder_path = f"uploads/{source_id}"
            os.makedirs(folder_path, exist_ok=True)

            filename = f"{int(time.time())}_{file.filename}"
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "wb") as f:
                f.write(contents)

            # Process and index
            processor = PDFProcessor([file_path])
            chunks = processor.process_pdfs()

            db = VectorDB(source_id=source_id)
            db.build_index(chunks)

        return {"message": "PDF uploaded and processed successfully", "source_id": source_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        if not request.source_id:
            raise HTTPException(status_code=400, detail="Missing source_id. Please provide one.")

        source_id = request.source_id
        question = request.question

        # Check cache
        if cached := cache.get_answer(question):
            return {"answer": cached, "source": "cache"}

        # Build path from source_id
        vector_db = VectorDB(source_id=source_id)

        if not vector_db.is_processed():
            return {"answer": "No documents have been processed yet.", "source": "system"}

        context, score = vector_db.get_relevant_context_with_score(question)
        SCORE_THRESHOLD = 0.3
        
        if context and score >= SCORE_THRESHOLD:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. Answer the user's question using only the context. "
                        "Do not mention or refer to the document. If context lacks info, say you couldn't find more details."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
            source = "document"
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Give a short answer or definition. "
                        "Say you couldn't find more details if needed."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            source = "openai"

        # Call OpenAI
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or fallback to "gpt-3.5-turbo"
            messages=messages
        )
        answer = response.choices[0].message.content.strip()

        cache.add_entry(question, answer)

        return {"answer": answer, "source": source}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

