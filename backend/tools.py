"""
    This tools working correctly
"""

import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rag_with_memory import MemoryRAG
import glob
from loguru import logger

possible_paths = [
    "/app/data/knowledge_base",
    "./data/knowledge_base",
    "./backend/data/knowledge_base"
]

KNOWLEDGE_BASE_PATH = None
for p in possible_paths:
    # Check if path exists AND contains .md files
    if os.path.exists(p) and glob.glob(os.path.join(p, "*.md")):
        KNOWLEDGE_BASE_PATH = p
        break

if not KNOWLEDGE_BASE_PATH:
    logger.critical("No .md files found in any knowledge base path!")
    rag_engine = None

else:
    logger.info(f"Knowledge Base detected at: {KNOWLEDGE_BASE_PATH}")
    try:
        rag_engine = MemoryRAG(docs_path=KNOWLEDGE_BASE_PATH)
        logger.success("RAG Engine initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize MemoryRAG: {e}")
        rag_engine = None

class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="User's question about coffee products, resets, warranty, installation safety, maintenance procedures, or troubleshooting guide.")

@tool(args_schema=KnowledgeBaseInput, return_direct=True)
def knowledge_base_search(query: str) -> str:
    """Search product documentation and FAQs to provide accurate answers about company products, technical procedures, warranty details, and maintenance schedules."""
    
    if not rag_engine:
        logger.warning(f"Search attempted but RAG engine is None. Query: {query}")
        return "I'm sorry, my internal knowledge base is currently offline. Please contact human support."

    try:
        result = rag_engine.query(query, session_id="agent_tool_session")
        return result.get("answer", "I couldn't find specific information about that in our records.")
    
    except Exception as e:
        # 2. Log the exact error for you to fix later
        logger.error(f"Error during RAG query: {e}")
        return "I encountered a technical error while searching the documents. Please try rephrasing."








'''
    Updated working tool with more time efficiency
'''
# import os
# import glob
# from langchain_core.tools import tool
# from pydantic import BaseModel, Field
# from rag_with_memory import get_rag_instance

# # 1. Path Discovery Logic
# possible_paths = [
#     "/app/data/knowledge_base",
#     "./data/knowledge_base",
#     "./backend/data/knowledge_base"
# ]

# KNOWLEDGE_BASE_PATH = None
# for p in possible_paths:
#     if os.path.exists(p) and glob.glob(os.path.join(p, "*.md")):
#         KNOWLEDGE_BASE_PATH = p
#         break

# # 2. Singleton Initialization
# # This runs once when the worker starts.
# if not KNOWLEDGE_BASE_PATH:
#     print("CRITICAL: No .md files found in any knowledge base path!")
#     rag_engine = None
# else:
#     print(f"Found Knowledge Base at: {KNOWLEDGE_BASE_PATH}")
#     try:
#         # Use the singleton getter to load the database once
#         rag_engine = get_rag_instance(docs_path=KNOWLEDGE_BASE_PATH)
#     except Exception as e:
#         print(f"Error initializing Local RAG: {e}")
#         rag_engine = None

# # 3. Tool Schema
# class KnowledgeBaseInput(BaseModel):
#     query: str = Field(description="User's question about coffee products, resets, or warranty")

# # 4. The Tool Definition
# @tool(args_schema=KnowledgeBaseInput, return_direct=True)
# def knowledge_base_search(query: str) -> str:
#     """Search product documentation and FAQs to provide accurate answers about company products, warranty, and reset procedures."""
    
#     # Safety check if initialization failed
#     if not rag_engine:
#         return "The knowledge base is currently unavailable."
    
#     # We call query() on the singleton instance.
#     # Note: 'rag_engine' is the instance returned by get_rag_instance()
#     result = rag_engine.query(query, session_id="agent_tool_session")
    
#     return result.get("answer", "No relevant information found in the documentation.")
