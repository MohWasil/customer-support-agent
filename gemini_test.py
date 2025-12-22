import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("Gemini_API")
# configure with your API key
genai.configure(api_key=API_KEY)

# pick a free model
model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Hello from Gemini! Explain what a coffee maker is.")
print("Response:", response.text)
