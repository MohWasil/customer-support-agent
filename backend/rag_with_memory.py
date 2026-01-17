"""
    RAG with Memory for customer support agent, main rag python file for searching and agent memory.    
"""

import os
import sys
from typing import Dict
from loguru import logger
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

load_dotenv()
# Setup production logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", level="INFO")

class MemoryRAG:
    def __init__(self, docs_path: str, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.docs_path = docs_path
        self.store: Dict[str, BaseChatMessageHistory] = {}
        
        try:
            logger.info(f"Initializing RAG with knowledge base: {docs_path}")
            
            # 1. Load and chunk documents
            loader = DirectoryLoader(docs_path, glob="*.md")
            docs = loader.load()
            if not docs:
                logger.warning(f"No documents found in {docs_path}. RAG will be empty.")

            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            # 2. Vector DB - Persistent storage
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.db = Chroma.from_documents(
                chunks, 
                embeddings, 
                persist_directory="./chroma_db"
            )

            # 3. LLM Setup
            hf_token = os.getenv("HF_API_TOKEN")
            if not hf_token:
                logger.critical("HF_API_TOKEN is missing from environment variables!")
                raise RuntimeError("HF_API_TOKEN not set")

            self.raw_llm = HuggingFaceEndpoint(
                repo_id=model,
                huggingfacehub_api_token=hf_token,
                temperature=0.1,
                max_new_tokens=200,
                return_full_text=False, 
                task="conversational"
            )
            self.llm = ChatHuggingFace(llm=self.raw_llm)

            # 4. Chains Setup
            self.retriever = self.db.as_retriever(search_kwargs={"k": 6})
            
            contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
            context_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

            history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, context_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are the SmartCoffee Support AI. Use the provided context to answer the user's question. "
                    "\n\n"
                    "### FORMATTING RULES:\n"
                    "- Use **Markdown** for all responses.\n"
                    "- If the answer involves a process or multiple steps, use a **numbered list** (1, 2, 3).\n"
                    "- If the answer contains several facts, use **bullet points** (•).\n"
                    "- Use **bold text** for button names or important terms (e.g., 'Press the **Brew** button').\n"
                    "- Keep the response concise and avoid long paragraphs."
                    "- If the answer is not in the context, say: 'I'm sorry, I don't have that specific policy in my records.'\n"
                    "- DO NOT use your internal knowledge to invent support tiers, response times, or phone numbers.\n"
                    "\n\n"
                    "Context: {context}"
                )),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                    ])
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            logger.success("MemoryRAG system initialized successfully.")

        except Exception as e:
            logger.exception("Failed to initialize MemoryRAG components")
            raise e

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def query(self, question: str, session_id: str = "default_session") -> dict:
        # Create a logger tied to this session
        session_logger = logger.bind(session_id=session_id)
        
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        try:
            session_logger.info(f"RAG Query received: {question[:50]}...")
            
            result = conversational_rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )

            # Extract sources directly from the result
            sources = list(set([doc.metadata.get("source", "unknown") for doc in result.get("context", [])]))

            session_logger.success("RAG Query completed.")
            return {
                "answer": result["answer"].strip(),
                "sources": sources
            }

        except Exception as e:
            session_logger.error(f"RAG Query Error: {e}")
            return {
                "answer": "I'm sorry, I encountered an error accessing my knowledge base.",
                "sources": []
            }
if __name__ == "__main__":
    rag = MemoryRAG("./backend/data/knowledge_base", model="meta-llama/Llama-3.1-8B-Instruct")
