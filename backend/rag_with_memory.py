# # backend/rag_with_memory.py
# from langchain_classic.memory import ConversationBufferWindowMemory
# # from langchain_classic.chains import ConversationRetrievalChain
# from langchain_classic.chains import conversational_retrieval
# # from langchain_classic.llms import HuggingFaceEndpoint
# # from langchain_classic.llms import huggingface_endpoint
# from langchain_huggingface.llms import HuggingFaceEndpoint
# from langchain_classic.vectorstores import Chroma
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_classic.embeddings import HuggingFaceEmbeddings
# from langchain_classic.prompts import PromptTemplate
# from langchain_classic.document_loaders import DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os

# class MemoryRAG:
#     def __init__(self, docs_path: str):
#         # 1. Load and chunk documents
#         loader = DirectoryLoader(docs_path, glob="*.md")
#         docs = loader.load()
        
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=50
#         )
#         chunks = splitter.split_documents(docs)
        
#         # 2. Vector DB
#         embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2"
#         )
#         self.db = Chroma.from_documents(
#             chunks,
#             embeddings,
#             persist_directory="./chroma_db"
#         )
        
#         # 3. LLM (HF API)
#         self.llm = HuggingFaceEndpoint(
#             repo_id="meta-llama/Llama-3.1-8B-Instruct",
#             huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
#             temperature=0.1,
#             max_new_tokens=200
#         )
        
#         # 4. Memory - keeps last 3 exchanges
#         self.memory = ConversationBufferWindowMemory(
#             memory_key="chat_history",
#             return_messages=True,
#             k=3
#         )
        
#         # 5. Custom prompt
#         prompt_template = """You are SmartCoffee Support AI. Use context and chat history.
        
#         Context: {context}
        
#         Chat History: {chat_history}
        
#         Question: {question}
        
#         Answer in 2-3 sentences. Be helpful but concise.
        
#         Answer:"""
        
#         PROMPT = PromptTemplate(
#             template=prompt_template,
#             input_variables=["context", "chat_history", "question"]
#         )
        
#         # 6. Conversational Chain
#         self.qa_chain = conversational_retrieval.from_llm(
#             llm=self.llm,
#             retriever=self.db.as_retriever(search_kwargs={"k": 2}),
#             memory=self.memory,
#             combine_docs_chain_kwargs={"prompt": PROMPT}
#         )
    
#     def query(self, question: str, session_id: str = None) -> dict:
#         # LangChain memory is not thread-safe, so we create per-session memory
#         # For now, use a simple session tracking (we'll improve this later)
        
#         result = self.qa_chain.invoke({
#             "question": question,
#             "chat_history": self.memory.chat_memory.messages
#         })
        
#         return {
#             "answer": result["answer"],
#             "sources": [doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])]
#         }

# # Test memory
# if __name__ == "__main__":
#     rag = MemoryRAG("./backend/data/knowledge_base")
    
#     # First question
#     print("Q1: How do I reset my coffee maker?")
#     r1 = rag.query("How do I reset my coffee maker?")
#     print(f"A1: {r1['answer']}\n")
    
#     # Follow-up question (should remember context)
#     print("Q2: What if I hold the button for too long?")
#     r2 = rag.query("What if I hold the button for too long?")
#     print(f"A2: {r2['answer']}")








# backend/rag_with_memory.py
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.documents import Document  
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# # from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains.retrieval import create_retrieval_chain
# from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_huggingface.llms import HuggingFaceEndpoint
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os

# class MemoryRAG:
#     def __init__(self, docs_path: str, model: str = "HuggingFaceTB/SmolLM2-360M"): # Updated model
#         # 1. Load and chunk documents
#         loader = DirectoryLoader(docs_path, glob="*.md")
#         docs = loader.load()
        
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=50
#         )
#         chunks = splitter.split_documents(docs)
        
#         # 2. Vector DB
#         embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2"
#         )
#         self.db = Chroma.from_documents(
#             chunks,
#             embeddings,
#             persist_directory="./chroma_db"
#         )
        
#         # 3. LLM (HF API)
#         self.llm = HuggingFaceEndpoint(
#             repo_id=model,
#             huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
#             temperature=0.1,
#             max_new_tokens=200,
#             return_full_text=False, # Important for cleaner output
#         )

#         # 4. Create the retriever
#         retriever = self.db.as_retriever(search_kwargs={"k": 2})

#         # 5. Create the history-aware retriever prompt
#         # This prompt helps the LLM reformulate the user's question considering the chat history
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages([
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ])

#         # 6. Create the history-aware retriever
#         history_aware_retriever = create_history_aware_retriever(
#             self.llm, retriever, contextualize_q_prompt
#         )

#         # 7. Create the main QA chain prompt
#         qa_system_prompt = """You are SmartCoffee Support AI. Use context and chat history.
        
#         Context: {context}
        
#         Chat History: {chat_history}

#         Answer in 2-3 sentences. Be helpful but concise."""
        
#         qa_prompt = ChatPromptTemplate.from_messages([
#             ("system", qa_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ])

#         # 8. Create the chain to generate answers from documents
#         question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

#         # 9. Combine the history-aware retriever and the QA chain
#         self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#         # 10. Manage sessions using a dictionary (not thread-safe for production)
#         self.store = {} # Dictionary to store chat histories per session ID

#     def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
#         """Retrieve or create a chat history for a given session ID."""
#         if session_id not in self.store:
#             self.store[session_id] = ChatMessageHistory()
#         return self.store[session_id]

#     def query(self, question: str, session_id: str = "default_session") -> dict:
#         """
#         Query the RAG system with memory.
#         Uses session_id to track conversation history.
#         """
#         # Wrap the RAG chain with message history management
#         conversational_rag_chain = RunnableWithMessageHistory(
#             self.rag_chain,
#             self.get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer",
#         )

#         try:
#             # Invoke the chain
#             result = conversational_rag_chain.invoke(
#                 {"input": question},
#                 config={
#                     "configurable": {"session_id": session_id}
#                 },
#             )

#             # Extract answer and sources
#             answer_text = result["answer"].strip()
            
#             # Note: Retrieval chain doesn't automatically return source docs like old chains
#             # To get sources, you'd need to inspect the intermediate steps or modify the chain
#             # For simplicity here, we'll return an empty list or you can modify the chain to pass through docs
#             # Let's assume source_documents are available somehow, maybe by modifying the chain further.
#             # A common way is to add a step that captures docs. For now, let's get them from the history-aware retriever's output.
#             # Actually, create_retrieval_chain does return 'context' which contains the documents used.
#             # We can extract metadata from the context if needed, but it's a list of strings by default.
#             # To get the actual Document objects with metadata, we might need a custom step.
#             # Let's simplify and just return the context content for now, or assume no direct source doc access easily.
#             # To get source docs, we'd ideally want the output of the retriever step within the chain.
#             # A workaround is to run the history_aware_retriever separately before the full chain if needed.
#             # For now, let's attempt to get the context from the result.
#             # print(result.keys()) # Uncomment to see what keys are available
#             # print(result.get('context', [])) # Uncomment to see context
            
#             # Getting sources is tricky with the new chains. The 'context' in the result is a list of strings.
#             # To get the original Document objects with metadata, we'd need to reconstruct or capture from the retriever.
#             # Let's try to access the retrieved documents indirectly if possible.
#             # A cleaner way involves customizing the chain creation further, which is more complex.
#             # For now, let's assume we don't get source documents easily with this new pattern,
#             # OR we can run the history aware retriever independently to get docs.
            
#             # Option 1: Assume no easy source docs with this setup (simplest)
#             # sources = []
            
#             # Option 2: Get docs by running the history_aware_retriever separately
#             # This is inefficient as it retrieves twice, but demonstrates getting docs.
#             # history = self.get_session_history(session_id)
#             # intermediate_result = history_aware_retriever.invoke({
#             #     "input": question,
#             #     "chat_history": history.messages
#             # })
#             # retrieved_docs = intermediate_result # This should be the list of documents
#             # sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]
            
#             # Option 3: Modify the chain creation to pass through docs (requires deeper customization)
#             # Let's go with Option 1 for simplicity, acknowledging the limitation.
            
#             # Let's try a simple attempt to see if context holds structured info by default
#             # context_str_list = result.get('context', [])
#             # print(f"Context type: {type(context_str_list)}, Content: {context_str_list[:2]}") # Log to understand
#             # Sources are difficult to extract cleanly without modification. 
#             # For this example, we'll return an empty list or comment on the challenge.
#             # sources = [] # Placeholder if we cannot get them easily
#             # Or, let's run the retriever step separately to get the actual docs with metadata:
#             history = self.get_session_history(session_id)
#             retrieved_docs_for_sources = self.db.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(
#                  question
#              ) # This doesn't consider history!
             
#              # To get docs considering history, we need the history_aware_retriever output:
#             history_aware_input = {
#                  "input": question,
#                  "chat_history": history.messages
#              }
#             # We cannot directly invoke history_aware_retriever with this input format easily outside the chain.
#             # The best way is to customize the chain creation to capture the output of the retriever step.
#             # For now, let's just get docs *without* history context for source listing, which is not ideal but可行.
#             docs_without_history = self.db.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(question)
#             sources = [doc.metadata.get("source", "unknown") for doc in docs_without_history]
            
#             # A better long-term solution would involve creating a custom chain component to capture the retriever output.

#             return {
#                 "answer": answer_text,
#                 "sources": sources
#             }
#         except Exception as e:
#             print(f"An error occurred during query execution: {e}")
#             return {"answer": f"An error occurred: {e}", "sources": []}


# # Test memory
# if __name__ == "__main__":
#     # Use a model confirmed to work with text generation, e.g., gemma-2-2b-it
#     rag = MemoryRAG("./backend/data/knowledge_base", model="HuggingFaceH4/zephyr-7b-beta") 

#     # First question
#     print("Q1: How do I reset my coffee maker?")
#     r1 = rag.query("How do I reset my coffee maker?", session_id="test_session_1")
#     print(f"A1: {r1['answer']}")
#     print(f"S1 Sources: {r1['sources']}\n")
    
#     # # Follow-up question (should remember context)
#     # print("Q2: What if I hold the button for too long?")
#     # r2 = rag.query("What if I hold the button for too long?", session_id="test_session_1")
#     # print(f"A2: {r2['answer']}")
#     # print(f"S2 Sources: {r2['sources']}\n")

#     # # New session
#     # print("Q3 (New Session): How do I clean the water tank?")
#     # r3 = rag.query("How do I clean the water tank?", session_id="test_session_2")
#     # print(f"A3: {r3['answer']}")
#     # print(f"S3 Sources: {r3['sources']}\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# backend/rag_with_memory.py
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.documents import Document
# from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains.retrieval import create_retrieval_chain
# # Change import for LLM

# from langchain_huggingface import HuggingFaceEndpoint
# # from langchain_huggingface import HuggingFaceHub # Use HuggingFaceHub instead of HuggingFaceEndpoint
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os

# class MemoryRAG:
#     def __init__(self, docs_path: str, model: str = "Qwen/Qwen2.5-7B-Instruct"): # Model that might be conversational
#         # 1. Load and chunk documents
#         loader = DirectoryLoader(docs_path, glob="*.md")
#         docs = loader.load()
        
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=50
#         )
#         chunks = splitter.split_documents(docs)
        
#         # 2. Vector DB
#         embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2"
#         )
#         self.db = Chroma.from_documents(
#             chunks,
#             embeddings,
#             persist_directory="./chroma_db"
#         )
        
        
#         self.llm = HuggingFaceEndpoint(
#             repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
#             # provider="together",
#             huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
#             task="conversational", # Explicitly set task
#             temperature=0.1,        # Pass directly
#             max_new_tokens=200,     # Pass directly
#             return_full_text=False, # Crucial: Return only the generated part
#         )

#         # 4. Create the retriever
#         retriever = self.db.as_retriever(search_kwargs={"k": 2})

#         # --- Prompt Templates ---
#         # For the history-aware retriever, it often works with text prompts
#         # but the LLM will handle the conversational aspect internally when needed.
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages([
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ])

#         # 5. Create the history-aware retriever
#         history_aware_retriever = create_history_aware_retriever(
#             self.llm, retriever, contextualize_q_prompt
#         )

#         # 6. Create the main QA chain prompt
#         # This prompt is passed to the LLM within the stuff_documents_chain.
#         # For a conversational model, structuring this as a system message followed by the question is often good.
#         qa_system_prompt = """You are SmartCoffee Support AI. Use context and chat history.
        
#         Context: {context}
        
#         Chat History: {chat_history}

#         Answer in 2-3 sentences. Be helpful but concise."""
        
#         qa_prompt = ChatPromptTemplate.from_messages([
#             ("system", qa_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"), # The user's question after history
#         ])

#         # 7. Create the chain to generate answers from documents
#         # This chain takes the context and reformatted input/question and generates an answer.
#         question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

#         # 8. Combine the history-aware retriever and the QA chain
#         self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#         # 9. Manage sessions
#         self.store = {}

#     def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
#         if session_id not in self.store:
#             self.store[session_id] = ChatMessageHistory()
#         return self.store[session_id]

#     def query(self, question: str, session_id: str = "default_session") -> dict:
#         conversational_rag_chain = RunnableWithMessageHistory(
#             self.rag_chain,
#             self.get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer",
#         )

#         try:
#             result = conversational_rag_chain.invoke(
#                 {"input": question},
#                 config={
#                     "configurable": {"session_id": session_id}
#                 },
#             )

#             answer_text = result["answer"].strip()
            
#             # Source retrieval challenge remains the same as before
#             history = self.get_session_history(session_id)
#             docs_without_history = self.db.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(question)
#             sources = [doc.metadata.get("source", "unknown") for doc in docs_without_history]

#             return {
#                 "answer": answer_text,
#                 "sources": sources
#             }
#         except Exception as e:
#             print(f"An error occurred during query execution: {e}")
#             return {"answer": f"An error occurred: {e}", "sources": []}


# # Test memory
# if __name__ == "__main__":
#     # Using the Qwen model which should be treated as conversational
#     rag = MemoryRAG("./backend/data/knowledge_base", model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

#     print("Q1: How do I reset my coffee maker?")
#     r1 = rag.query("How do I reset my coffee maker?", session_id="test_session_1")
#     print(f"A1: {r1['answer']}")
#     print(f"S1 Sources: {r1['sources']}\n")
    
#     # print("Q2: What if I hold the button for too long?")
#     # r2 = rag.query("What if I hold the button for too long?", session_id="test_session_1")
#     # print(f"A2: {r2['answer']}")
#     # print(f"S2 Sources: {r2['sources']}\n")

#     # print("Q3 (New Session): How do I clean the water tank?")
#     # r3 = rag.query("How do I clean the water tank?", session_id="test_session_2")
#     # print(f"A3: {r3['answer']}")
#     # print(f"S3 Sources: {r3['sources']}\n")







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

        # 3. LLM: use the model argument, and pass model kwargs
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
#     ### New updates comes on to reduce time latency
#     _rag_instance = None

#     def get_rag_instance(docs_path: str = "./backend/data/knowledge_base"):
#         global _rag_instance
#         if _rag_instance is None:
#             print("Initializing MemoryRAG for the first time (Loading ChromaDB)...")
#             _rag_instance = MemoryRAG(docs_path)
#         return _rag_instance
    


    
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
