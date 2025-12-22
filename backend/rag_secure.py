# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from session_manager import SessionManager
# from langchain_classic.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain_classic.prompts import PromptTemplate
# from langchain_classic.document_loaders import DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# load_dotenv()
# import os

# class SecureRAG:
#     def __init__(self, docs_path: str):
#         # Creating instance of sessionmanager
#         self.session_manager = SessionManager()
#         # Setup vector DB (same as before)
#         loader = DirectoryLoader(docs_path, glob="*.md")
#         docs = loader.load()
#         chunks = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=50
#         ).split_documents(docs)
        
#         embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2"
#         )
#         self.db = Chroma.from_documents(
#             chunks,
#             embeddings,
#             persist_directory="./chroma_db"
#         )
        
#         # LLM
#         llm_endpoint = HuggingFaceEndpoint(
#             repo_id="meta-llama/Llama-3.1-8B-Instruct",
#             huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
#             temperature=0.1,
#             max_new_tokens=200,
#             task="conversational"
#         )
#         self.llm = ChatHuggingFace(llm=llm_endpoint)
#         # Prompt
#         self.prompt = PromptTemplate(
#             template="""You are SmartCoffee Support AI. Use context and chat history.
            
#             Context: {context}
            
#             Chat History: {chat_history}
            
#             Question: {question}
            
#             Answer in 2-3 sentences. Be helpful but concise.
            
#             Answer:""",
#             input_variables=["context", "chat_history", "question"]
#         )
    
#     def query(self, question: str, session_id: str) -> dict:
#         # Get session-specific memory
#         memory = self.session_manager.get_or_create_session(session_id=session_id)
#         # self.llm = ChatHuggingFace(llm=self.llm)
#         # Build chain with session memory
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=self.db.as_retriever(search_kwargs={"k": 2}),
#             memory=memory,
#             combine_docs_chain_kwargs={"prompt": self.prompt}
#         )
        
#         result = qa_chain.invoke({"question": question})
        
#         return {
#             "answer": result["answer"],
#             "sources": [doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])],
#             "session_id": session_id
#         }

# # Singleton
# secure_rag = SecureRAG("./backend/data/knowledge_base")









# rag_secure.py
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from session_manager import SessionManager # Assuming this handles chat history correctly
# Use modern imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
import os

class SecureRAG:
    def __init__(self, docs_path: str):
        # Setup vector DB (same as before)
        loader = DirectoryLoader(docs_path, glob="*.md")
        docs = loader.load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        ).split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="./chroma_db"
        )

        # LLM - Use HuggingFaceEndpoint directly for text generation/conversation
        self.llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct", # Use a model confirmed for text-gen
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
            temperature=0.1,
            max_new_tokens=200,
            # task="text-generation", # Often optional, defaults correctly
            return_full_text=False,
        )
        self.llm = ChatHuggingFace(llm=self.llm)
        # Create the retriever
        retriever = self.db.as_retriever(search_kwargs={"k": 2})

        # --- History-Aware Retriever Prompt ---
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

        # Create the history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # --- Main QA Prompt ---
        qa_system_prompt = """You are SmartCoffee Support AI. Use context and chat history.

        Context: {context}

        Chat History: {chat_history}

        Answer in 2-3 sentences. Be helpful but concise."""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Create the chain to generate answers from documents
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # Combine the history-aware retriever and the QA chain
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Manage sessions using a dictionary (not thread-safe for production)
        self.store = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def query(self, question: str, session_id: str) -> dict:
        # Wrap the RAG chain with message history management
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        try:
            result = conversational_rag_chain.invoke(
                {"input": question},
                config={
                    "configurable": {"session_id": session_id}
                },
            )

            answer_text = result["answer"].strip()
            # Source retrieval is complex with create_retrieval_chain, simplified here
            # Docs without history for source listing (not ideal for memory context)
            docs_without_history = self.db.as_retriever(search_kwargs={"k": 2}).invoke(question)
            sources = [doc.metadata.get("source", "unknown") for doc in docs_without_history]

            return {
                "answer": answer_text,
                "sources": sources,
                "session_id": session_id
            }
        except Exception as e:
            print(f"Error in SecureRAG.query: {e}")
            return {
                "answer": f"An error occurred retrieving the answer: {e}",
                "sources": [],
                "session_id": session_id
            }

# Singleton
secure_rag = SecureRAG("./data/knowledge_base")