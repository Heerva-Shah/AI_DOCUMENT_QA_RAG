# backend/main.py

import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from ingestion import ingest_document
from retrieval import answer_question, is_multiple_questions
from memory import clear_history

app = FastAPI(title="AI Document Q&A System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Track last generated content type per session (in-memory, simple)
session_last_generated = {}

class ChatRequest(BaseModel):
    question: str
    session_id: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    session_id = str(uuid.uuid4())
    clear_history(session_id)
    session_last_generated[session_id] = None

    try:
        result = ingest_document(file_bytes, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to process document. Please try again.")

    return {
        "session_id": result["session_id"],
        "num_chunks": result["num_chunks"],
        "message": "Document uploaded and processed successfully."
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required. Please upload a document first.")

    if is_multiple_questions(request.question):
        raise HTTPException(status_code=400, detail="Please ask one question at a time for best results.")

    last_type = session_last_generated.get(request.session_id)

    try:
        result = answer_question(request.question, request.session_id, last_type)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to generate answer. Please try again.")

    # Update tracked state for next message
    session_last_generated[request.session_id] = result.get("generated_type")

    return {"answer": result["answer"]}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)