"""
    Schema File to make sure clean the prompts 
"""
from pydantic import BaseModel, Field, field_validator
import re
import time
from typing import List

class ChatRequest(BaseModel):
    # Standardizing question length for model performance and cost control
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="The user's query for the AI agent"
    )
    
    # Enhanced pattern for common prefixes like 'http_'
    session_id: str = Field(
        default="default", 
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        max_length=64
    )
    
    @field_validator('question')
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        # 1. Normalize whitespace
        v = re.sub(r'\s+', ' ', v).strip()
        
        # 2. Advanced Security: Heuristic check for prompt injection
        forbidden_patterns = [
            r"ignore previous instructions", 
            r"system prompt", 
            r"reveal your secrets",
            r"new instructions",
            r"you are now an admin"
        ]
        
        lower_v = v.lower()
        for pattern in forbidden_patterns:
            if re.search(pattern, lower_v):
                raise ValueError("Message contains restricted administrative patterns.")
        
        return v

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    session_id: str
    timestamp: float = Field(default_factory=time.time)
