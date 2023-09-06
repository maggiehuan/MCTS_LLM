import os
import requests
import json

model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
model_choice = "gpt-35-turbo"
API_KEY = os.environ.get("OPENAI_API_KEY")
print(API_KEY)
#API_ENDPOINT = "https://api.openai.com/v1/engines/davinci-codex/completions"
API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"
# prompt = "which country is the richest and why"
with open("/home/ziyu/code/LLMs/mcts-llm/src/tot/data/crosswords/mini0505.json", "r") as file:
    prompt = json.load(file)
input = prompt[0][0]
print(input)

input_data = {  
    "messages": [  
        {"role": "system", "content": "Solve 5x5 mini crosswords. Given an input of 5 horizontal clues and 5 vertical clues, generate thoughts about which 5-letter word fits each clue, then an output of 5 rows, where each row is 5 letter separated by space. The first one is an example."},  
        {"role": "user", "content": "Input: h1. A lunar valley\nh2. A fatty oil\nh3. To entice\nh4. To lower; to reduce\nh5. A solitary person\nv1. According to the roster\nv2. Another name for Port-Francqui\nv3. An illicit lover; a European lake\nv4. To lisp\nv5. To come in\nThoughts:\nh1. Presented; revealed: SHOWN\nh2. An interjection expressing sorrow: WIRRA\nh3. Benefit; result: AVAIL\nh4. A cigarette: RETTE\nh5. Chased up a tree: TREED\nv1. Swarthy; tawny: SWART\nv2. An apiarist or bee keeper: HIVER\nv3. To speak formally: ORATE\nv4. To indite; to scribble: WRITE\nv5. An insecticide: NALED\nOutput:R I L L E\nO L E I N\nT E M P T\nA B A S E\nL O N E R\n Input: {}\n".format(input)},
        #{"role": "user", "content": "Output:"}
    ],  
    "max_tokens": 500,
    "temperature": 0.7,
    "n":5
}

headers = {'Content-Type': 'application/json', 'api-key': API_KEY}  
response = requests.post(API_ENDPOINT, json=input_data, headers=headers)

# action=[]
# for i in range(5):
#     content = response['message']['content']
#     output_start = content.find("Output:") + len("Output:")
#     output = content[output_start:].strip()   
#     action.append(output[0])


outputs = []

for choice in response['choices']:
    content = choice['message']['content']

    import re
    output_match = re.search(r'Output:(.*?)(Input:|$)', content, re.DOTALL)
    if output_match:
        output = output_match.group(1).strip()
        outputs.append(output)

print(outputs)

# for i in range(5):
#     response[i]
#     passage = response[i].json()
# Split the passage into lines using '\n' as the delimiter
    # lines = passage.split('\n')

# Check if there are at least two newline characters in the passage
    # if len(lines) >= 3:  # We check for >= 3 because the first line is empty
        # Extract the text after the second '\n' (newline character)
    #     extracted_text = lines[2]
    #     print(extracted_text)
    # else:
    #     print("There are not enough newline characters in the passage.")


# output = response.choices[0].message["content"]
print(response.json())
# if response.status_code == 200:
#     result = response.json()
#     #generated_text = result["choices"][0]["text"]
#     #genertaed_text = result["choices"][3]["content"]
#     generated_text = result["choices"][3]["content"]
#     print(genertaed_text)
# else:
#     print("Error:", response.text)

#API_KEY = os.environ.get("OPENAI_API_KEY")
