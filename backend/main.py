from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from file_handler import load_and_split
from rag_pipeline import build_vectorstore, answer_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    file_bytes = await file.read()
    chunks = load_and_split(file_bytes, file.filename)
    build_vectorstore(chunks)
    return {"message": f"File '{file.filename}' processed successfully.", "chunks": len(chunks)}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer = answer_question(request.question)
    return AnswerResponse(answer=answer)