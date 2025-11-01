# backend/debug.py
from __future__ import annotations
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/debug", tags=["debug"])

_last_plan: Dict[str, Any] = {}

def set_last_plan(plan: Dict[str, Any]):
    global _last_plan
    _last_plan = plan or {}

@router.get("/sql")
def get_last_sql():
    # Returns the most recent plan captured during /ask
    return _last_plan or {"note": "No plan recorded yet."}
