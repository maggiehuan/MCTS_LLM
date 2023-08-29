import os
import requests

API_KEY = os.environ.get("OPENAI_API_KEY")
print(API_KEY)
#API_ENDPOINT = "https://api.openai.com/v1/engines/davinci-codex/completions"
API_ENDPOINT = 'https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview'
prompt = "which country is the richest and why"

data = {
    "prompt": prompt,
    "max_tokens": 100
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.post(API_ENDPOINT, json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    generated_text = result["choices"][0]["text"]
    print(generated_text)
else:
    print("Error:", response.text)
import os

API_KEY = os.environ.get("OPENAI_API_KEY")
