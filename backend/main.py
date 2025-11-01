# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from .planner import answer_question
from fastapi.responses import RedirectResponse

app = FastAPI(title="Samarth ")

class QueryIn(BaseModel):
    q: str

@app.post("/ask")
def ask(body: QueryIn) -> Dict[str, Any]:
    try:
        result = answer_question(body.q)
        return result
    except Exception as e:
        return {"answer_text": f"Error: {e}", "tables": [], "citations": []}

@app.get("/health")
def health():
    return {"status": "ok"}

# Serve the frontend at /ui (avoid shadowing /ask)
from fastapi.staticfiles import StaticFiles
from pathlib import Path
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")




@app.get("/")
def root():
    return RedirectResponse(url="/ui/")
