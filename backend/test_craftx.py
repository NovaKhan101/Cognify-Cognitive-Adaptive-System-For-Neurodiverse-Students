# test_craftx.py
import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("CRAFTX_API_KEY")
ENDPOINT = "https://api.craftx.corecraftsolutions.com/api/v1/chat/completions"

response = requests.post(
    ENDPOINT,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "Gemini 2.5 Flash",
        "messages": [
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
)

data = response.json()
print("Response:", data["choices"][0]["message"]["content"])
print("Tokens used:", data["usage"]["total_tokens"])
if "cost_bdt" in data:
    print(f"Cost: ৳{data['cost_bdt']}")
    print(f"Balance: ৳{data['balance_after']}")
if data.get("warning"):
    print(f"WARNING: {data['warning']}")