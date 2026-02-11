import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROK_API_KEY") # User uses GROK_API_KEY
if not key:
    print("No API Key found!")
    exit(1)

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 10
}

try:
    print(f"Testing key: {key[:10]}...")
    resp = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
