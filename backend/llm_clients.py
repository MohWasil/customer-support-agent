# import os
# from groq import Groq
# from huggingface_hub import InferenceClient
# from monitoring import record_agent_metrics  
# from typing import Optional
# from dotenv import load_dotenv
# import time
# load_dotenv()

# class LLMClient:
#     """Switch between HF API and Groq based on availability"""
    
#     def __init__(self):
#         self.hf_token = os.getenv("HF_API_TOKEN")
#         self.groq_token = os.getenv("Grouq_API_KEY")
        
#         # Initialize Groq client
#         self.groq_client = Groq(api_key=self.groq_token) if self.groq_token else None
        
#         # Model mapping
#         self.primary_model = "groq" if self.groq_token else "huggingface"
        
        
#     def generate(self, prompt: str, max_tokens: int = 300) -> dict:
#         start_time = time.time()
#         status = "success"
        
#         try:
#             if self.primary_model == "groq":
#                 result = self._groq_generate(prompt, max_tokens, start_time)
#             else:
#                 result = self._hf_generate(prompt, max_tokens, start_time)
            
#             # --- RECORD METRICS ---
#             record_agent_metrics(
#                 model=result["model"],
#                 latency=result["latency"],
#                 tokens_in=result["tokens_in"],
#                 tokens_out=result["tokens_out"],
#                 status=status
#             )
#             return result
                
#         except Exception as e:
#             status = "error"            
#             # Fallback logic
#             if self.primary_model == "huggingface" and self.groq_client:
#                 print("HF failed, falling back to Groq")
#                 return self._groq_generate(prompt, max_tokens, start_time)
#             raise e

    
#     def _groq_generate(self, prompt: str, max_tokens: int, start_time: float) -> dict:
#         response = self.groq_client.chat.completions.create(
#             model="llama3-8b-8192",  
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=max_tokens,
#             temperature=0.1
#         )
        
#         latency = time.time() - start_time
        
#         return {
#             "text": response.choices[0].message.content,
#             "latency": latency,
#             "tokens_in": response.usage.prompt_tokens,
#             "tokens_out": response.usage.completion_tokens,
#             "model": "groq-llama3-8b"
#         }
    
#     def _hf_generate(self, prompt: str, max_tokens: int, start_time: float) -> dict:
#         client = InferenceClient(
#         model="meta-llama/Llama-3.1-8B-Instruct",
#         token=self.hf_token)
    
#         start_time = time.time()
        
#         # Use chat_completion for conversational history
#         response = client.chat_completion(
#             messages=prompt, 
#             max_tokens=max_tokens,
#             temperature=0.1
#         )
        
#         latency = time.time() - start_time
        
#         # Get REAL token usage from the response object
#         tokens_in = response.usage.prompt_tokens
#         tokens_out = response.usage.completion_tokens
        
#         return {
#             "text": response.choices[0].message.content,
#             "latency": latency,
#             "tokens_in": tokens_in,
#             "tokens_out": tokens_out,
#             "model": "hf-llama3.1-8b"
#         }
        
#         # from huggingface_hub import InferenceClient
        
#         # client = InferenceClient(
#         #     model="meta-llama/Llama-3.1-8B-Instruct",
#         #     token=self.hf_token
#         # )
        
#         # response = client.text_generation(
#         #     prompt,
#         #     max_new_tokens=max_tokens,
#         #     temperature=0.1
#         # )
        
#         # latency = time.time() - start_time
        
#         # # HF API doesn't return token counts, estimate
#         # est_tokens_in = len(prompt.split()) * 1.3
#         # est_tokens_out = len(response.split()) * 1.3
        
#         # return {
#         #     "text": response,
#         #     "latency": latency,
#         #     "tokens_in": int(est_tokens_in),
#         #     "tokens_out": int(est_tokens_out),
#         #     "model": "hf-llama3.1-8b"
#         # }