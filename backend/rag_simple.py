# from langchain_community.vectorstores.chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from huggingface_hub import InferenceClient
# # from langchain.document_loaders import DirectoryLoader
# from langchain_classic.document_loaders import DirectoryLoader
# from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from dotenv import load_dotenv
# load_dotenv()



# # google/gemma-2b-it
# class SimpleRAG:
#     def __init__(self, docs_path: str, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
#         # 1. Load documents
#         loader = DirectoryLoader(docs_path, glob="*.md")
#         docs = loader.load()
#         print(f"Loaded {len(docs)} documents")

#         # 2. Chunk them
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=50
#         )
#         chunks = splitter.split_documents(docs)
#         print(f"Created {len(chunks)} chunks")

#         # 3. Embeddings
#         print("Downloading embeddings model (one-time)...")
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


#         self.db = Chroma.from_documents(
#             chunks,
#             embeddings,
#             persist_directory="./chroma_db"
#         )
#         print("Vector DB ready")

#         # 5. HF Inference client + model id (pass model to text_generation, not constructor)
        
#         self.model = model

#         self.client = InferenceClient(
#             model= self.model,
#             token=os.getenv("HF_API_TOKEN")
#                 )

#     def _extract_generated_text(self, resp):
#         # try to handle common response shapes returned by different huggingface_hub versions
#         if isinstance(resp, str):
#             return resp
#         if isinstance(resp, dict):
#             if "generated_text" in resp:
#                 return resp["generated_text"]
#             if "text" in resp:
#                 return resp["text"]
#             if "outputs" in resp and isinstance(resp["outputs"], list) and isinstance(resp["outputs"][0], dict):
#                 return resp["outputs"][0].get("generated_text") or resp["outputs"][0].get("text") or str(resp["outputs"][0])
#             # fallback
#             return str(resp)
#         if isinstance(resp, list) and resp:
#             return self._extract_generated_text(resp[0])
#         return str(resp)

#     def query(self, question: str) -> dict:
#         # Retrieve top 2 docs
#         docs = self.db.similarity_search(question, k=2)
#         context = "\n\n".join([d.page_content for d in docs])

#         # Build prompt
#         prompt = f"""You are a customer support agent. Use ONLY the context below.

#         Context:
#         {context}

#         Question: {question}

#         Answer based ONLY on the context. If unsure, say "I need to check that with my team."
#         Keep answer under 3 sentences.

#         Answer:"""

#         # Generate -> pass model explicitly and use `inputs=` + parameters
#         resp = self.client.text_generation(
#             prompt,
#             # model=self.model,
#             max_new_tokens=150,
#             temperature= 0.1
#         )

#         answer_text = self._extract_generated_text(resp).strip()

#         # Extract sources
#         sources = [{"source": d.metadata.get("source", "unknown")} for d in docs]

#         return {"answer": answer_text, "sources": sources}


# # Test it
# if __name__ == "__main__":
#     rag = SimpleRAG("./backend/data/knowledge_base")

#     test_questions = [
#         "How do I reset my coffee maker?"
#     ]

#     for q in test_questions:
#         print(f"\nQuestion: {q}")
#         result = rag.query(q)
#         print(f"Answer: {result['answer']}")
#         print(f"Sources: {[s['source'] for s in result['sources']]}")













from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

# Qwen/Qwen2.5-1.5B-Instruct
# microsoft/phi-3.5-mini-3.8b-instruct
# microsoft/DialoGPT-medium
# Using a conversational model like DialoGPT
class SimpleRAG:
    def __init__(self, docs_path: str, model: str = "google/gemma-2-2b-it"): # Conversational model
        # 1. Load documents
        loader = DirectoryLoader(docs_path, glob="*.md")
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")

        # 2. Chunk them
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks")

        # 3. Embeddings
        print("Downloading embeddings model (one-time)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="./chroma_db"
        )
        print("Vector DB ready")

        # 5. HF Inference client
        self.model = model
        self.client = InferenceClient(
            model=self.model,
            token=os.getenv("HF_API_TOKEN")
        )

    # Note: DialoGPT might need simpler history management, this is adapted for chat_completion structure
    # DialoGPT usually works better with previous conversation context as input
    def query(self, question: str) -> dict:
        # Retrieve top 2 docs
        docs = self.db.similarity_search(question, k=2)
        context = "\n\n".join([d.page_content for d in docs])

        # Build prompt for a conversational model
        # DialoGPT expects a history of conversation. We'll format context as a system/user message.
        # However, HuggingFace's chat_completion might expect a list of message dictionaries.
        # Let's try the standard chat format first.
        # Note: DialoGPT might not handle system messages well; if this fails, we might need text_generation or a different model.
        messages = [
            {"role": "system", "content": f"You are a customer support agent. Use ONLY the context below to answer the user's question. Context: {context}"},
            {"role": "user", "content": question}
        ]

        try:
            # Use chat_completion for conversational models
            resp = self.client.chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=0.1
            )

            # Extract the generated text from the response
            # The structure might vary slightly depending on the exact model/provider
            # Typical structure: {'choices': [{'message': {'content': '...'}}]}
            if resp and resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                 answer_text = resp.choices[0].message.content.strip()
            else:
                print(f"Unexpected response structure: {resp}")
                answer_text = "Could not extract a valid response from the model."

            # Extract sources
            sources = [{"source": d.metadata.get("source", "unknown")} for d in docs]

            return {"answer": answer_text, "sources": sources}

        except StopIteration:
            print(f"StopIteration error encountered with model {self.model}. The model might be temporarily unavailable or misconfigured on the inference endpoint.")
            return {"answer": "An error occurred processing your request (StopIteration).", "sources": []}
        except Exception as e:
            print(f"An unexpected error occurred with model {self.model}: {e}")
            # Attempt to get a generic response or return an error message
            # For DialoGPT-like models, `text_generation` might be more appropriate, but requires different prompt formatting
            return {"answer": f"An unexpected error occurred with model {self.model}. Error: {e}", "sources": []}


# Test it
if __name__ == "__main__":
    rag = SimpleRAG("./backend/data/knowledge_base")

    test_questions = [
        "How do I reset my coffee maker?"
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        result = rag.query(q)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {[s['source'] for s in result['sources']]}")
