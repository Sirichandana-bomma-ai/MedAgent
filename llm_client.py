# llm_client.py
import os
import requests

# Make sure you have set your Together API key:
TOGETHER_API_KEY="599e27e1942e9b8dca5563d56ffcc5a6c8e4b03c7af15f62fa95bed863637885"

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def run_llm(user_prompt: str, model: str, system_prompt: str = None):
    """Call Together API using REST instead of Python SDK"""
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY not found. Please set it as an environment variable.")

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4000
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]