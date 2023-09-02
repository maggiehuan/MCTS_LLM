import os
import requests
import json

model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
model_choice = "gpt-4-32k"
API_KEY = os.environ.get("OPENAI_API_KEY")
print(API_KEY)
#API_ENDPOINT = "https://api.openai.com/v1/engines/davinci-codex/completions"
API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"
# prompt = "which country is the richest and why"
with open("/home/ziyu/code/LLMs/mcts-llm/src/tot/data/crosswords/mini0505.json", "r") as file:
    prompt = json.load(file)
# data = {
#     "messages": prompt,
#     "max_tokens": 100
# }
descriptions = prompt[0]

input_data = {  
    "messages": [  
        {"role": "system", "content": "Solve 5x5 mini crosswords. Given an input of 5 horizontal clues and 5 vertical clues, generate thoughts about which 5-letter word fits each clue, then an output of 5 rows, where each row is 5 letter separated by space."},  
        {"role": "user", "content": "Input: h1. A lunar valley\nh2. A fatty oil\nh3. To entice\nh4. To lower; to reduce\nh5. A solitary person\nv1. According to the roster\nv2. Another name for Port-Francqui\nv3. An illicit lover; a European lake\nv4. To lisp\nv5. To come in\nThoughts:\nh1. Presented; revealed: SHOWN\nh2. An interjection expressing sorrow: WIRRA\nh3. Benefit; result: AVAIL\nh4. A cigarette: RETTE\nh5. Chased up a tree: TREED\nv1. Swarthy; tawny: SWART\nv2. An apiarist or bee keeper: HIVER\nv3. To speak formally: ORATE\nv4. To indite; to scribble: WRITE\nv5. An insecticide: NALED\nOutput:R I L L E\nO L E I N\nT E M P T\nA B A S E\nL O N E R\nInput:\nAn agendum; something to be done\nAn engine\nPretentious; flowery\nA salon; a hall\nTo mock; to sneer\nTo heap\nAn Indian antelope\nTo intend; to plan; to devise; a nettle; to guess\nA nozzle\nDesiccator; more dry"},
        #{"role": "user", "content": "Output:"}
    ],  
    "max_tokens": 500,
    "temperature": 0.7,
    "n":5
}

# for description in descriptions:
#     input_data["messages"].append({"role": "user", "content": description})

# input_json = json.dumps(input_data, indent=4)

# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {API_KEY}",
#     "api-key": api_key
# }
headers = {'Content-Type': 'application/json', 'api-key': API_KEY}  
response = requests.post(API_ENDPOINT, json=input_data, headers=headers)

output = response.choices[0].message["content"]
print(output)
# if response.status_code == 200:
#     result = response.json()
#     #generated_text = result["choices"][0]["text"]
#     #genertaed_text = result["choices"][3]["content"]
#     generated_text = result["choices"][3]["content"]
#     print(genertaed_text)
# else:
#     print("Error:", response.text)

#API_KEY = os.environ.get("OPENAI_API_KEY")
