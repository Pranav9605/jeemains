# backend/app/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PDFProcessResponse(BaseModel):
    status: str
    error: Optional[str] = None
    num_questions: Optional[int] = None

class QARequest(BaseModel):
    question: str
    k: int = 3

class QAResponse(BaseModel):
    predicted_answer: str
    confidence: float
    supporting_questions: List[Dict[str, Any]]
