"""
    Best Schema version been used
"""

# from pydantic import BaseModel, Field, field_validator
# import re
# import time

# class ChatRequest(BaseModel):
#     # Field(...) is used for required fields; 
#     # 'pattern' replaces V1's 'regex'
#     question: str = Field(..., min_length=1, max_length=500)
#     session_id: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]+$")
    
#     @field_validator('question')
#     @classmethod
#     def sanitize_question(cls, v: str) -> str:
#         # Normalize whitespace
#         v = re.sub(r'\s+', ' ', v).strip()
        
#         # Check for potential malicious patterns
#         if any(word in v.lower() for word in ['ignore', 'system', 'admin']):
#             raise ValueError("Potentially malicious pattern detected")
#         return v

# class ChatResponse(BaseModel):
#     question: str
#     answer: str
#     sources: list
#     session_id: str
#     # Added a default_factory for timestamp to automate it
#     timestamp: float = Field(default_factory=time.time)











"""
    Updated the Schema for production
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
    
    # Enhanced pattern for 2026 session management (allowing common prefixes like 'http_')
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
        # Instead of just keywords, we check for 'system instructions' logic
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
                # In production, we raise a 422 error via FastAPI
                raise ValueError("Message contains restricted administrative patterns.")
        
        return v

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    session_id: str
    timestamp: float = Field(default_factory=time.time)
