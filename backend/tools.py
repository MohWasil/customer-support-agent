# # backend/tools.py
# from langchain_core.tools import tool
# from typing import Optional, Type
# from pydantic import BaseModel, Field

# class KnowledgeBaseInput(BaseModel):
#     query: str = Field(description="User's question about coffee products")

# # class KnowledgeBaseInput(BaseModel):
# #     query: str = Field(description="The search query to look up in the documentation")

# @tool(args_schema=KnowledgeBaseInput)
# def knowledge_base_search(query: str) -> str:
#     """Search product documentation and FAQs to provide accurate answers about the company products and policies."""
#     from rag_secure import secure_rag
    
#     # Implementation logic
#     # Note: Using a tool-specific session ID as in your previous code
#     result = secure_rag.query(query, session_id="system_tool")
#     return result["answer"]


from langchain_core.tools import tool
from pydantic import BaseModel, Field

class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="User's question about coffee products")

@tool(args_schema=KnowledgeBaseInput, return_direct=True)
def knowledge_base_search(query: str) -> str:
    """Search product documentation and FAQs to provide accurate answers about the company products and policies."""
    from rag_secure import secure_rag
    result = secure_rag.query(query, session_id="system_tool")
    return result["answer"]

