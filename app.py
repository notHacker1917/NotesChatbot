from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rag_engine import query_notes

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str
    k: int = 3

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = query_notes(request.query, k=request.k)
    return {"response": result}
