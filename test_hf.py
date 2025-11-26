# # Create test script
# from huggingface_hub import InferenceClient
# import os
# from dotenv import load_dotenv
# load_dotenv()

# def test_hf_api():
#     # Load token safely
#     token = os.getenv("HF_API_TOKEN")
#     if not token:
#         print("HF_API_TOKEN not found in .env")
#         return False
    
#     try:
#         client = InferenceClient(
#             model="Qwen/Qwen2.5-1.5B-Instruct",
#             token=token
#         )
        
#         print("Calling HF API (this may take 5-10s on first run)...")
#         response = client.text_generation(
#             "What is 2+2?",
#             max_new_tokens=50,
#             temperature=0.1
#         )
        
#         print(f"Success! Response: {response[:100]}...")
#         return True
        
#     except Exception as e:
#         print(f"Error: {e}")
#         return False

# if __name__ == "__main__":
#     test_hf_api()



# import os
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_openai import ChatOpenAI
# api_key = os.getenv("Feather_API_KEY")

# llm = ChatOpenAI(
#     api_key=api_key,
#     base_url="https://api.featherless.ai/v1",
#     model="meta-llama/Llama-3.1-8B-Instruct")

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     (
#         "human",
#         "I love programming."
#     ),
# ]
# ai_msg = llm.invoke(messages)
# ai_msg    






import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Define Model ID ---
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# --- 2. Load Tokenizer and Model ---
# The 'token' parameter uses your logged-in access token automatically
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency (if supported by your GPU, e.g., Ampere architecture or newer)
    device_map="auto",          # Automatically uses GPU if available, otherwise CPU
    token=True                  # Authenticates the request
)

# --- 3. Prepare Input Prompt ---
# Use the chat template for instruction-following models
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the concept of quantum entanglement in simple terms."}
]

# Apply the template and convert messages to input IDs
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# --- 4. Generate Response ---
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response_ids = outputs[0][input_ids.shape[-1]:]
response = tokenizer.decode(response_ids, skip_special_tokens=True)

# --- 5. Print the Output ---
print("Model Response:")
print(response)
