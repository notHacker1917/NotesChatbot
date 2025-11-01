from dotenv import load_dotenv
import os

load_dotenv()  # loads .env
api_key = os.getenv("OPENAI_API_KEY")
print("Your API key is:", api_key[:5] + "..." if api_key else "Not found")
