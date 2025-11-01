from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import query_notes

app = FastAPI(title="Notes Chatbot")

# Allow frontend (any origin for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Notes Chatbot is running!"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    if not question:
        return JSONResponse(content={"answer": "No question provided."}, status_code=400)
    
    answer = query_notes(question)
    return {"answer": answer}
