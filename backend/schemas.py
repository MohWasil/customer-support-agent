# from pydantic import BaseModel, Field, validator
# import re

# class ChatRequest(BaseModel):
#     question: str = Field(..., min_length=3, max_length=500)
#     session_id: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]+$")
    
#     @validator('question')
#     def sanitize_question(cls, v):
#         # Remove potential prompt injection patterns
#         v = re.sub(r'\s+', ' ', v).strip()  # Normalize whitespace
#         if any(word in v.lower() for word in ['ignore', 'system', 'admin']):
#             raise ValueError("Potentially malicious pattern detected")
#         return v

# class ChatResponse(BaseModel):
#     question: str
#     answer: str
#     sources: list
#     session_id: str
#     timestamp: float


from pydantic import BaseModel, Field, field_validator
import re
import time

class ChatRequest(BaseModel):
    # Field(...) is used for required fields; 
    # 'pattern' replaces V1's 'regex'
    question: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]+$")
    
    @field_validator('question')
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        # Normalize whitespace
        v = re.sub(r'\s+', ' ', v).strip()
        
        # Check for potential malicious patterns
        if any(word in v.lower() for word in ['ignore', 'system', 'admin']):
            raise ValueError("Potentially malicious pattern detected")
        return v

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list
    session_id: str
    # Added a default_factory for timestamp to automate it
    timestamp: float = Field(default_factory=time.time)
