'''
Docstring for backend.rag_with_memory
This version worked perfectly.
'''


import os
import traceback
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
from pathlib import Path

# Initalizing the .env path
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class MemoryRAG:
    def __init__(self, docs_path: str, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        # 1. Load and chunk documents
        loader = DirectoryLoader(docs_path, glob="*.md")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # 2. Vector DB
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

        # 3. LLM: use the model argument
        hf_token = os.getenv("HF_API_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_API_TOKEN environment variable not set")

        # Use HuggingFaceHub wrapper which expects repo_id and model_kwargs
        self.llm = HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=hf_token,
            temperature=0.1,
            max_new_tokens=200,
            return_full_text=False, 
            task="conversational"
        )
        self.llm = ChatHuggingFace(llm=self.llm)
        # 4. Retriever
        retriever = self.db.as_retriever(search_kwargs={"k": 2})

        # 5. Prompt templates
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

        qa_system_prompt = """You are SmartCoffee Support AI. Use context and chat history.

Context: {context}

Chat History: {chat_history}

Answer in 2-3 sentences. Be helpful but concise."""
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        self.store = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def query(self, question: str, session_id: str = "default_session") -> dict:
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        try:
            # invoke and capture full exception if fails
            result = conversational_rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )

            answer_text = result["answer"].strip()
            docs_without_history = self.db.as_retriever(search_kwargs={"k": 2}).invoke(question)
            sources = [doc.metadata.get("source", "unknown") for doc in docs_without_history]

            return {"answer": answer_text, "sources": sources}

        except Exception as e:
            print("Full traceback:\n", traceback.format_exc())
            return {"answer": f"An error occurred: {e}", "sources": []}


# Quick test
if __name__ == "__main__":
    rag = MemoryRAG("./backend/data/knowledge_base", model="meta-llama/Llama-3.1-8B-Instruct")
    # print("Q1: How do I reset my coffee maker?")
    # r1 = rag.query("How do I reset my coffee maker?", session_id="test_session_1")
    # print("A1:", r1["answer"])
    # print("Sources:", r1["sources"])











# '''
#     New Version of the code.
# '''
# import os
# import traceback
# from pathlib import Path
# from dotenv import load_dotenv

# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains.retrieval import create_retrieval_chain

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Initialize environment
# env_path = Path(__file__).resolve().parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)

# class MemoryRAG:
#     def __init__(self, docs_path: str, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
#         self.model_name = model
        
#         # 1. Vector DB Logic (Singleton friendly)
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         persist_dir = "./chroma_db"

#         # Only process documents if the DB doesn't exist to save time
#         if not os.path.exists(persist_dir):
#             print("Processing documents for new Vector DB...")
#             loader = DirectoryLoader(docs_path, glob="*.md")
#             docs = loader.load()
#             splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
#             chunks = splitter.split_documents(docs)
#             self.db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
#         else:
#             print("Loading existing Vector DB from disk...")
#             self.db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

#         # 2. LLM Setup
#         hf_token = os.getenv("HF_API_TOKEN")
#         if not hf_token:
#             raise RuntimeError("HF_API_TOKEN environment variable not set")

#         endpoint = HuggingFaceEndpoint(
#             repo_id=model,
#             huggingfacehub_api_token=hf_token,
#             temperature=0.1,
#             max_new_tokens=200,
#             return_full_text=False, 
#             task="conversational"
#         )
#         self.llm = ChatHuggingFace(llm=endpoint)

#         # 3. Chains Setup
#         retriever = self.db.as_retriever(search_kwargs={"k": 2})

#         # History Awareness
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "formulate a standalone question. Do NOT answer the question."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages([
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ])
#         history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

#         # QA Chain
#         qa_system_prompt = """You are SmartCoffee Support AI. Use context and chat history.
#         Context: {context}
#         Chat History: {chat_history}
#         Answer in 2-3 sentences."""
        
#         qa_prompt = ChatPromptTemplate.from_messages([
#             ("system", qa_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ])

#         question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
#         self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
#         self.store = {}

#     def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
#         if session_id not in self.store:
#             self.store[session_id] = ChatMessageHistory()
#         return self.store[session_id]

#     def query(self, question: str, session_id: str = "default_session", callbacks=None) -> dict:
#         """
#         Query the RAG chain. 
#         'callbacks' allows Prometheus MetricsCallbackHandler to track this specific run.
#         """
#         conversational_rag_chain = RunnableWithMessageHistory(
#             self.rag_chain,
#             self.get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer",
#         )

#         try:
#             # Pass the callbacks into the config
#             result = conversational_rag_chain.invoke(
#                 {"input": question},
#                 config={
#                     "configurable": {"session_id": session_id},
#                     "callbacks": callbacks
#                 },
#             )

#             answer_text = result["answer"].strip()
#             # Fast source retrieval
#             sources = [doc.metadata.get("source", "unknown") for doc in result.get("context", [])]

#             return {"answer": answer_text, "sources": sources}

#         except Exception as e:
#             print("Full traceback:\n", traceback.format_exc())
#             return {"answer": f"An error occurred: {e}", "sources": []}




# # --- SINGLETON PATTERN ---
# _rag_instance = None
# def get_rag_instance():
#     global _rag_instance
#     if _rag_instance is None:
#         print("Initializing MemoryRAG Singleton...")
#         _rag_instance = MemoryRAG(docs_path="./data/knowledge_base")
#     return _rag_instance


# # Quick test logic
# # if __name__ == "__main__":
# #     rag = get_rag_instance()
#     # print(rag.query("How do I reset my coffee machine?")["answer"])
