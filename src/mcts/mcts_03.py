import numpy as np
import json
from src.tot.prompts.crosswords import cot_prompt
from src.tot.models import gpt
from src.mcts.crossword_mcts import CrosswordsEnv
from src.tot.models import gpt

import random
import os
import requests
import json
import re
from collections import defaultdict

model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
model_choice = "gpt-4-32k"
API_KEY = os.environ.get("OPENAI_API_KEY")
API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"

prompt_path = "/home/ziyu/code/LLMs/mcts-llm/src/tot/data/crosswords/mini0505.json"


input = json.load(open(prompt_path))
input = input[5][0] # 循环的i改在第一个框内

env_crosswords = CrosswordsEnv(prompt_path)
# input = env_crosswords.prompt_input()

# input_data = {  
#     "messages": [  
#         {"role": "system", "content": "Solve 5x5 mini crosswords. Given an input of 5 horizontal clues and 5 vertical clues, generate thoughts about which 5-letter word fits each clue, then an output of 5 rows, where each row is 5 letter separated by space."},  
#         {"role": "user", "content": "Input: h1. A lunar valley\nh2. A fatty oil\nh3. To entice\nh4. To lower; to reduce\nh5. A solitary person\n\
# v1. According to the roster\nv2. Another name for Port-Francqui\nv3. An illicit lover; a European lake\nv4. To lisp\nv5. To come in\n\
# Thoughts:\nh1. Presented; revealed: SHOWN\nh2. An interjection expressing sorrow: WIRRA\nh3. Benefit; result: AVAIL\nh4. A cigarette: RETTE\nh5. Chased up a tree: TREED\n\
# v1. Swarthy; tawny: SWART\nv2. An apiarist or bee keeper: HIVER\nv3. To speak formally: ORATE\nv4. To indite; to scribble: WRITE\nv5. An insecticide: NALED\n\
# Output:R I L L E\nO L E I N\nT E M P T\nA B A S E\nL O N E R\n\
# Input: {input}\n".format(input=input)},
#         #{"role": "user", "content": "Output:"}
#     ],  
#     "max_tokens": 500,
#     "temperature": 0.7,
#     "n":5
# }

headers = {'Content-Type': 'application/json', 'api-key': API_KEY}  
# response = requests.post(API_ENDPOINT, json=input_data, headers=headers)


N = 10  # Rollouts
L = 5   # Depth
num_actions = 3  
C = 1.0  
learning_rate = 0.1  

# Dictionaries of Q and N values
Q = {}
N_count = {}

# Rollout process
def rollout(env, s, depth):
    if depth == 0:
        return 0

    if s in Q:
        a_best = max(range(num_actions), key=lambda a: Q[(s, a)] + C * np.sqrt(np.log(N_count[s]) / N_count[(s, a)]))
    else:
        a_best = np.random.choice(num_actions)
        Q[(s, a_best)] = 0
        N_count[(s, a_best)] = 0

    N_count[s] += 1
    s_new = s + a_best
    if s_new == "answered":
        reward = env.reward(s_new)  
        update_value(s, reward)
    else:
        reward = rollout(s_new, depth - 1)

    # Backpropagation
    update_value(s, reward)
    return reward


def update_value(state, reward, a_best, Q, N_count):
    Q[(state, a_best)] = (Q[(state, a_best)] * N_count[(state, a_best)] + reward) / (N_count[(state, a_best)] + 1)
    N_count[(state, a_best)] += 1


# history = env1.prompt_wrap
a_star = []

def get_best_action(self):
    best_child = max(self.children, key=lambda node: node.visits)
    return self.get_action_to_reach_child(best_child)

def PUCT(Q, N, t, c=1.0):
    # PUCT 
    values = np.array([Q[a] + c * np.sqrt(np.log(t) / (N[a] + 1)) for a in Q])
    chosen_action = np.argmax(values)
    return chosen_action

def possible_actions(env, state):
    response = requests.post(API_ENDPOINT, json=env.get_input_data(state), headers=headers)
    response_exm = response.json()
    # print(response.json())
    unique_actions = []
    num_actions = 0
    # TODO 这里需要删掉所有的/r和/n, 然后把相同的句子分类 done
    for i in range(6):
        content = response_exm['choices'][i]['message']['content']
        content = content.split('\n')
        if content[0] not in unique_actions:
            #unique_actions.append(content)
            unique_actions = content + unique_actions
            num_actions += 1
        # print(content)
    print(unique_actions)
    return unique_actions, num_actions
    #content = response_exm['choices'][i]['message']['content']

    # unique_actions = []
    # unique_actions = defaultdict(content)
    #print(unique_actions)
    # for i, content in enumerate(unique_actions):
    #     print(content)
    
    # actions = []
    #print(response.json())
    #response_exm = {'id': 'chatcmpl-7vgfmUUFqFgezspU4okLbr00iM71w', 'object': 'chat.completion', 'created': 1693983086, 'model': 'gpt-4-32k', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'h1. An agendum; something to be done: TASKS\nh2. An engine: MOTOR\nh3. Pretentious; flowery: FANCY\nh4. A salon; a hall: PARLOR\nh5. To mock; to sneer: SCOFF\n\nv1. To heap: STACK\nv2. An Indian antelope: NILGAI\nv3. To intend; to plan; to devise; a nettle; to guess: MEANT\nv4. A nozzle: SPOUT\nv5. Desiccator; more dry: DRIER\n\nOutput: T A S K S\nM O T O R\nF A N C Y\nP A R L O R\nS C O F F'}, 'finish_reason': 'stop'}, {'index': 1, 'message': {'role': 'assistant', 'content': 'h1. An agendum; something to be done: TASKS\nh2. An engine: MOTOR\nh3. Pretentious; flowery: FANCY\nh4. A salon; a hall: PARLOR\nh5. To mock; to sneer: SCOFF\n\nv1. To heap: PILE\nv2. An Indian antelope: NILGAI\nv3. To intend; to plan; to devise; a nettle; to guess: MEANT\nv4. A nozzle: SPOUT\nv5. Desiccator; more dry: DRIER\n\nOutput: T A S K S\nM O T O R\nF A N C Y\nP A R L O R\nS C O F F'}, 'finish_reason': 'stop'}, {'index': 2, 'message': {'role': 'assistant', 'content': 'h1. An agendum; something to be done: TASKS\r\nh2. An engine: MOTOR\r\nh3. Pretentious; flowery: FANCY\r\nh4. A salon; a hall: LOBBY\r\nh5. To mock; to sneer: SCOFF\r\nv1. To heap: PILET\r\nv2. An Indian antelope: NILGA\r\nv3. To intend; to plan; to devise; a nettle; to guess: MEANT\r\nv4. A nozzle: SPOUT\r\nv5. Desiccator; more dry: DRIER\r\nOutput:\r\nT A S K S\r\nM O T O R\r\nF A N C Y\r\nL O B B Y\r\nS C O F F'}, 'finish_reason': 'stop'}, {'index': 3, 'message': {'role': 'assistant', 'content': 'h1. An agendum; something to be done: TASKS\r\nh2. An engine: MOTOR\r\nh3. Pretentious; flowery: FANCY\r\nh4. A salon; a hall: FOYER\r\nh5. To mock; to sneer: SCOFF\r\nv1. To heap: STACK\r\nv2. An Indian antelope: NILGA\r\nv3. To intend; to plan; to devise; a nettle; to guess: MEANT\r\nv4. A nozzle: SPOUT\r\nv5. Desiccator; more dry: DRIER\r\nOutput: T A S K S\r\nM O T O R\r\nF A N C Y\r\nF O Y E R\r\nS C O F F'}, 'finish_reason': 'stop'}, {'index': 4, 'message': {'role': 'assistant', 'content': 'h1. An agendum; something to be done: TASKS\r\nh2. An engine: MOTOR\r\nh3. Pretentious; flowery: FANCY\r\nh4. A salon; a hall: PARLOR\r\nh5. To mock; to sneer: SCOFF\r\n\r\nv1. To heap: PILE\r\nv2. An Indian antelope: NILGAI\r\nv3. To intend; to plan; to devise; a nettle; to guess: MEANT\r\nv4. A nozzle: SPOUT\r\nv5. Desiccator; more dry: DRIER\r\n\r\nOutput:\r\nT A S K S\r\nM O T O R\r\nF A N C Y\r\nP A R L O R\r\nS C O F F'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 410, 'completion_tokens': 800, 'total_tokens': 1210}}
    # actions_dict = {"h1": [], "h2": [], "h3": [], "h4": [], "h5": []}

    #for i in range(5):
        # actions = [r.text for r in (response[i])]
        #actions_dict = {"h1": [], "h2": [], "h3": [], "h4": [], "h5": []}
        # action_dict = []
        #content = response_exm['choices'][i]['message']['content']
        # print(content)
        # print('------------------------------------------------------')
        #content = content.split('\n')
        # print(content)

        # for line in content:
        #     if line.startswith("h1."):
        #         #action_dict.append(line.strip())
        #         actions_dict["h1"].append(line.strip())
        #     elif line.startswith("h2."):
        #         #action_dict.append(line.strip())
        #         actions_dict["h2"].append(line.strip())
        #     elif line.startswith("h3."):
        #         #action_dict.append(line.strip())
        #         actions_dict["h3"].append(line.strip())
        #     elif line.startswith("h4."):
        #         #action_dict.append(line.strip())
        #         actions_dict["h4"].append(line.strip())
        #     elif line.startswith("h5."):
        #         #action_dict.append(line.strip())
        #         actions_dict["h5"].append(line.strip())
        # #print(actions_dict)



    # key_order = ["h1", "h2", "h3", "h4", "h5"]  # 定义键的顺序

    # for key in key_order:
    #     if key in actions_dict:
    #         #print(f'{key}:')
    #         for item in actions_dict[key]:
    #             print(item)
    # for key, value in actions_dict.items():
    #     for item in actions_dict[key]:
    #         print(item)
    # return actions_dict


def main():
#     # M_theta = GenerativeLLM()  
#     # optimizer = optim.Adam(M_theta.parameters(), lr=0.001)
    initial_state = env_crosswords.get_input_data()
    reward = 0.0
    N = 10  # Number of rollouts
    depth = 3  # Depth of exploration
    depth_limit = 5  # Depth limit
    iterations = 10  # Training iterations
    T = 5

    state_set = set()
    tree = {}

    while True:
        if state not in state_set:
            state_set.add(state)

        rewards = []

        for i in possible_actions(env_crosswords, state).count:
            
            current_chain = []

            unique_actions = []
            unique_actions[i] = possible_actions(env_crosswords, state)[i]

            # TODO 需要把生成过的unique action append到state里面？confused

            # 加到tree的chain里面 -> confused either
            if unique_actions[i] not in current_chain:
                current_chain.append(unique_actions[i])

                if state not in tree:
                    tree[state] = []
                
                tree[state].append(current_chain.copy())
  
            for current_state in state_set:
                update_value(current_state, rewards)

                new_chains = possible_actions(env_crosswords, current_state)

                if current_state not in tree:
                    tree[current_state] = []

                tree[current_state].extend(new_chains)

                reward = env_crosswords.reward(current_state)
                rewards.append(reward)


        # Q,V
        update_value(state, rewards)

        
        if env_crosswords.answered():
            break
    
        if state in state_set:
            actions = [current_state]
            
            best_action = get_best_action(Q, N, current_state)
            
            new_state = state + best_action
            
            if new_state is None:
                break
        
    update_value(state, reward, best_action, Q, N)
    current_state = new_state



if __name__ == "__main__":
    state = ""
    possible_actions(env_crosswords, state)
