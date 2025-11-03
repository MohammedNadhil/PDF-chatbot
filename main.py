from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import PDFProcessor
from vector_db import VectorDB
from multi_vector_db import MultiVectorDB
from semantic_cache import SemanticQuestionCache
from pydantic import BaseModel, validator
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import List, Optional, Tuple

import os
import hashlib
import time

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

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
    # Don't block startup with PDF processing
    import asyncio
    asyncio.create_task(preload_pdfs_async())

async def preload_pdfs_async():
    try:
        VectorDB(source_id="preloaded").preload_pdfs("preloaded_pdfs")
    except Exception as e:
        print(f"PDF preloading failed: {e}")

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "TTII Chatbot API is running"}

@app.get("/sources")
async def get_available_sources():
    """Get information about available processed sources"""
    try:
        # Get all source directories
        sources = []
        
        # Check processed_texts directory
        if os.path.exists("processed_texts"):
            for filename in os.listdir("processed_texts"):
                if filename.endswith('.json'):
                    source_id = filename.replace('.json', '')
                    vector_db = VectorDB(source_id=source_id)
                    sources.append({
                        "source_id": source_id,
                        "processed": vector_db.is_processed(),
                        "text_count": len(vector_db.texts) if hasattr(vector_db, 'texts') else 0
                    })
        
        return {
            "sources": sources,
            "total_sources": len(sources),
            "processed_sources": len([s for s in sources if s["processed"]])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup")
async def cleanup_storage():
    """Clean up old PDF files and unused processed data"""
    try:
        cleaned_files = []
        cleaned_folders = []
        
        # Clean up uploads directory
        if os.path.exists("uploads"):
            for folder_name in os.listdir("uploads"):
                folder_path = os.path.join("uploads", folder_name)
                if os.path.isdir(folder_path):
                    # Check if this source has been processed
                    source_id = folder_name
                    vector_db = VectorDB(source_id=source_id)
                    
                    if vector_db.is_processed():
                        # Source is processed, safe to delete PDF files
                        for file_name in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, file_name)
                            if file_name.endswith('.pdf'):
                                os.remove(file_path)
                                cleaned_files.append(file_path)
                        
                        # Remove empty folder
                        if not os.listdir(folder_path):
                            os.rmdir(folder_path)
                            cleaned_folders.append(folder_path)
        
        return {
            "message": "Cleanup completed",
            "cleaned_files": cleaned_files,
            "cleaned_folders": cleaned_folders,
            "total_files_removed": len(cleaned_files),
            "total_folders_removed": len(cleaned_folders)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChapterRelevanceRequest(BaseModel):
    chapter_title: str
    topic: str
    question: str
    source_id: str
    
    @validator('chapter_title')
    def validate_chapter_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Chapter title cannot be empty')
        return v.strip()
    
    @validator('topic')
    def validate_topic(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Topic cannot be empty')
        return v.strip()
    
    @validator('question')
    def validate_question(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Question cannot be empty')
        return v.strip()
    
    @validator('source_id')
    def validate_source_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Source ID cannot be empty')
        return v.strip()


class QuestionRequest(BaseModel):
    question: str
    source_ids: List[str] | None = None  # Support multiple sources
    source_id: str | None = None  # Keep for backward compatibility
    public: int = 1  # New parameter: 0=PDF only, 1=PDF+OpenAI fallback (default)
    
    @validator('source_ids')
    def validate_source_ids(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError('source_ids cannot be empty list')
        return v
    
    @validator('public')
    def validate_public(cls, v):
        if v not in [0, 1]:
            raise ValueError('public must be 0 or 1')
        return v

@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    try:
        processed_source_ids = []
        
        for file in files:
            contents = await file.read()
            source_id = hashlib.md5(contents).hexdigest()

            folder_path = f"uploads/{source_id}"
            os.makedirs(folder_path, exist_ok=True)

            filename = f"{int(time.time())}_{file.filename}"
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Save PDF temporarily for processing
                with open(file_path, "wb") as f:
                    f.write(contents)

                # Process and index
                processor = PDFProcessor([file_path])
                chunks = processor.process_pdfs()

                db = VectorDB(source_id=source_id)
                db.build_index(chunks)

                processed_source_ids.append(source_id)
                
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                continue
                
            finally:
                # Clean up: Delete PDF file and folder after processing
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    if os.path.exists(folder_path) and not os.listdir(folder_path):
                        os.rmdir(folder_path)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up files for {source_id}: {cleanup_error}")

        return {
            "message": "PDFs processed successfully.", 
            "source_ids": processed_source_ids,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def validate_context_relevance(question: str, context: str) -> bool:
    """Use OpenAI to validate if context actually answers the question"""
    try:
        validation_prompt = f"""
        Question: {question}
        Context: {context}
        
        IMPORTANT: Only mark as RELEVANT if the context contains SPECIFIC information about the exact topic asked.
        
        Examples:
        - Question: "What is JavaScript?" + Context about "JavaScript programming" = RELEVANT
        - Question: "What is JavaScript?" + Context about "mathematics and real numbers" = NOT_RELEVANT
        - Question: "What is PHP?" + Context about "PHP web development" = RELEVANT  
        - Question: "What is PHP?" + Context about "education and teaching" = NOT_RELEVANT
        
        Does this context contain specific information that directly answers the question?
        Answer: RELEVANT or NOT_RELEVANT
        """
        
        print(f"DEBUG: Validating relevance for question: {question}")
        print(f"DEBUG: Context preview: {context[:200]}...")
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": validation_prompt}]
        )
        
        result = response.choices[0].message.content.strip().upper()
        print(f"DEBUG: Validation result: {result}")
        
        # Fix: Check for exact match, not substring
        is_relevant = result == "RELEVANT"
        print(f"DEBUG: Is relevant: {is_relevant}")
        
        return is_relevant
    except Exception as e:
        print(f"Error in relevance validation: {e}")
        return False  # Default to not relevant if validation fails

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question

        # Handle both single source_id and multiple source_ids
        if request.source_ids:
            source_ids = request.source_ids
        elif request.source_id:
            source_ids = [request.source_id]
        else:
            # No source provided - use general knowledge
            source_ids = None

        # Check cache only for preloaded source
        if source_ids == ["preloaded"] and (cached := cache.get_answer(question)):
            return {"answer": cached, "source": "cache"}

        # Handle different source scenarios
        if source_ids is None:
            # No source provided - use general knowledge only
            context = None
            score = 0.0
            source_info = []
        elif len(source_ids) == 1:
            # Single source - use original VectorDB for efficiency
            vector_db = VectorDB(source_id=source_ids[0])
            
            if not vector_db.is_processed():
                return {"answer": "No documents have been processed yet.", "source": "system"}

            context, score = vector_db.get_relevant_context_with_score(question)
            source_info = [{"source_id": source_ids[0], "score": float(score), "context_length": len(context) if context else 0}]
        else:
            # Multiple sources - use MultiVectorDB
            multi_vector_db = MultiVectorDB(source_ids)
            
            if not multi_vector_db.is_any_processed():
                return {"answer": "No documents have been processed yet.", "source": "system"}

            context, score, source_info = multi_vector_db.get_relevant_context_from_all_sources(question)

        SCORE_THRESHOLD = 0.5  # Increased from 0.3 to be more strict
        
        # Check if we have context and it meets similarity threshold
        if context and score >= SCORE_THRESHOLD:
            print(f"DEBUG: Found context with score {score}, proceeding to relevance validation")
            # Additional validation: check if context is actually relevant to the question
            is_relevant = await validate_context_relevance(question, context)
            print(f"DEBUG: Relevance validation result: {is_relevant}")
            
            if is_relevant:
                print("DEBUG: Context is relevant, using document-based answer")
                # Document-based answer (context is relevant)
                if len(source_ids) == 1:
                    system_prompt = (
                        "You are a helpful AI assistant. Answer the user's question naturally and conversationally using the provided information. "
                        "Don't mention based on documents, context, or sources. Just give a direct, helpful answer as if you naturally know this information."
                    )
                    source = "document"
                else:
                    system_prompt = (
                        "You are a helpful AI assistant. Answer the user's question naturally and conversationally using the provided information. "
                        "Don't mention based ondocuments, context, or sources. Just give a direct, helpful answer as if you naturally know this information."
                    )
                    source = "multiple_documents"
                
            messages = [
                {
                    "role": "system",
                        "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        else:
                print("DEBUG: Context is not relevant, treating as no relevant context")
                # Context found but not relevant - treat as no relevant context
                context = None
                score = 0.0
        
        # Handle cases where no relevant context was found
        if not context or score < SCORE_THRESHOLD:
            print(f"DEBUG: No relevant context found (score: {score}, threshold: {SCORE_THRESHOLD})")
            # No relevant context found in PDFs
            if request.public == 1:
                print("DEBUG: Public mode - falling back to OpenAI general knowledge")
                # Public mode: fallback to OpenAI general knowledge
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant. Answer the user's question naturally and conversationally. "
                            "Provide a direct, helpful answer as if you naturally know this information. "
                            "Don't mention that you're an AI or that you're using general knowledge."
                        )
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
                source = "general"
            else:
                print("DEBUG: Private mode (public=0) - returning 'not enough information' message")
                # Private mode: only PDF responses, no fallback
                response_data = {
                    "answer": "I don't have enough information in the provided documents to answer this question accurately.",
                    "source": "system"
                }
                
                # Add source metadata based on the scenario
                if len(source_ids) == 1:
                    response_data["source_id"] = source_ids[0]
                elif len(source_ids) > 1:
                    response_data["processed_sources"] = multi_vector_db.get_processed_sources()
                
                return response_data

        # Call OpenAI
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer = response.choices[0].message.content.strip()

        # Only cache responses from preloaded source
        if source_ids == ["preloaded"]:
            cache.add_entry(question, answer)

        # Prepare response with source information
        response_data = {
            "answer": answer, 
            "source": source
        }
        
        # Add source metadata based on the scenario
        if source_ids is None:
            # No source provided - just return the answer
            pass
        elif len(source_ids) > 1:
            response_data["processed_sources"] = multi_vector_db.get_processed_sources()
        elif len(source_ids) == 1:
            response_data["source_id"] = source_ids[0]

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
async def get_sources():
    """Get list of available processed sources"""
    try:
        sources = []
        
        # Check processed_texts directory
        if os.path.exists("processed_texts"):
            for file in os.listdir("processed_texts"):
                if file.endswith('.json'):
                    source_id = file.replace('.json', '')
                    vector_db = VectorDB(source_id=source_id)
                    if vector_db.is_processed():
                        sources.append({
                            "source_id": source_id,
                            "text_count": len(vector_db.texts),
                            "is_processed": True
                        })
        
        return {
            "sources": sources,
            "total_sources": len(sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-relevance")
async def check_chapter_relevance(request: ChapterRelevanceRequest):
    """
    Check if a question is relevant to a specific chapter and topic.
    Returns a professional response with relevance indicator.
    """
    try:
        chapter_title = request.chapter_title
        topic = request.topic
        question = request.question
        source_id = request.source_id
        
        # Check if source exists and is processed
        vector_db = VectorDB(source_id=source_id)
        if not vector_db.is_processed():
            return {
                "answer": "The specified source has not been processed yet. Please upload and process documents first.",
                "is_relevant": 0,
                "source": "system"
            }
        
        # First, search for content related to the chapter title and topic
        chapter_topic_query = f"{chapter_title} {topic}"
        chapter_context, chapter_score = vector_db.get_relevant_context_with_score(chapter_topic_query)
        
        # Also search for content related to the question to see if it matches the chapter context
        question_context, question_score = vector_db.get_relevant_context_with_score(question)
        
        # STRICT VALIDATION: Only proceed if we have chapter-specific content
        # If no chapter content is found, the question cannot be relevant to this chapter
        if not chapter_context or chapter_score < 0.3:
            return {
                "answer": f"No content found for chapter '{chapter_title}' and topic '{topic}'. Please ensure you're asking about the correct chapter content.",
                "is_relevant": 0,
                "source": "system",
                "chapter_title": chapter_title,
                "topic": topic,
                "source_id": source_id
            }
        
        # Use the chapter context as primary for validation
        primary_context = chapter_context
        
        # Create a comprehensive prompt for relevance checking
        relevance_prompt = f"""
You are an educational content analyzer. Your task is to determine if a student's question is relevant to a specific chapter and topic from educational material.

Chapter Title: "{chapter_title}"
Topic: "{topic}"
Student's Question: "{question}"

Content from the specified Chapter/Topic area:
{chapter_context}

Content found for the student's question:
{question_context if question_context else "No relevant content found for this question"}

CRITICAL VALIDATION RULES:
1. The question MUST be directly related to the specified chapter title AND topic
2. The question content MUST match or be contained within the chapter/topic content
3. If the question is about a completely different subject (even if it exists elsewhere in the document), it is NOT_RELEVANT
4. Only mark as RELEVANT if the question is asking about concepts, definitions, or information that is explicitly covered in the chapter/topic content

Examples of NOT_RELEVANT scenarios:
- Question about "Carl Friedrich Gauss" when chapter is "Real Numbers" and topic is "Introduction" = NOT_RELEVANT
- Question about "JavaScript programming" when chapter is "Mathematics" and topic is "Algebra" = NOT_RELEVANT
- Question about "History of computers" when chapter is "Real Numbers" and topic is "Introduction" = NOT_RELEVANT

Examples of RELEVANT scenarios:
- Question about "What are real numbers?" when chapter is "Real Numbers" and topic is "Introduction" = RELEVANT
- Question about "How to add real numbers?" when chapter is "Real Numbers" and topic is "Operations" = RELEVANT

Respond with:
- "RELEVANT" ONLY if the question directly relates to the chapter and topic content
- "NOT_RELEVANT" if the question is unrelated to the chapter and topic OR if the content doesn't match

Format your response as:
RELEVANCE: [RELEVANT/NOT_RELEVANT]
RESPONSE: [Your educational response based on the chapter content]
"""
        
        # Call OpenAI for relevance analysis
        messages = [
            {
                "role": "system",
                "content": "You are an educational content analyzer. Analyze questions for relevance to specific chapters and topics."
            },
            {
                "role": "user",
                "content": relevance_prompt
            }
        ]
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        analysis = response.choices[0].message.content.strip()
        
        # Parse the response
        lines = analysis.split('\n')
        relevance_status = "NOT_RELEVANT"
        answer_text = analysis
        
        for line in lines:
            if line.startswith("RELEVANCE:"):
                relevance_status = line.split(":", 1)[1].strip()
            elif line.startswith("RESPONSE:"):
                answer_text = line.split(":", 1)[1].strip()
        
        # Determine relevance flag - be strict about matching
        is_relevant = 1 if relevance_status.upper() == "RELEVANT" else 0
        
        # If not relevant, provide a more professional message
        if is_relevant == 0:
            answer_text = f"This question is not relevant to the current chapter '{chapter_title}' and topic '{topic}'. Please ask about concepts covered in this chapter, such as: {topic.lower()}."
        
        return {
            "answer": answer_text,
            "is_relevant": is_relevant,
            "source": "document" if is_relevant else "system",
            "chapter_title": chapter_title,
            "topic": topic,
            "source_id": source_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=False)
