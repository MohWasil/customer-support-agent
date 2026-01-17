"""
    This tools File
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
        # Log the exact error
        logger.error(f"Error during RAG query: {e}")
        return "I encountered a technical error while searching the documents. Please try rephrasing."
