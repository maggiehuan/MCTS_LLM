import openai
import torch
import torch.optim as optim
import numpy as np
import json
from src.tot.prompts.crosswords import cot_prompt
from tot.models import gpt
from src.tot.tasks.crosswords import MiniCrosswordsEnv, CrosswordsEnv
from tot.models import gpt

import os
import requests
import json

model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
model_choice = "gpt-4-32k"
API_KEY = os.environ.get("OPENAI_API_KEY")
API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"

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

headers = {'Content-Type': 'application/json', 'api-key': API_KEY}  
# response = requests.post(API_ENDPOINT, json=input_data, headers=headers)

prompt_path = "/home/ziyu/code/LLMs/mcts-llm/src/tot/data/crosswords/mini0505.json"

env = MiniCrosswordsEnv()
env1 = CrosswordsEnv()

N = 10  # Rollouts
L = 5   # Depth
num_actions = 3  
C = 1.0  
learning_rate = 0.1  

# Dictionaries of Q and N values
Q = {}
N_count = {}

# Rollout process
def rollout(s, depth):
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
        reward = env1.reward(s_new)  
        update_value(s, reward)
    else:
        reward = rollout(s_new, depth - 1)

    # Backpropagation
    update_value(s, reward)
    return reward


def update_value(state, reward, a_best, Q, N_count):
    Q[(state, a_best)] = (Q[(state, a_best)] * N_count[(state, a_best)] + reward) / (N_count[(state, a_best)] + 1)
    N_count[(state, a_best)] += 1


history = env.prompt_status_cache
history = env1.prompt_wrap
a_star = []

def prompt_wrap(obs):
    return cot_prompt.format(input=obs)

def get_best_action(self):
    best_child = max(self.children, key=lambda node: node.visits)
    return self.get_action_to_reach_child(best_child)

def PUCT(Q, N, t, c=1.0):
    # PUCT 
    values = np.array([Q[a] + c * np.sqrt(np.log(t) / (N[a] + 1)) for a in Q])
    chosen_action = np.argmax(values)
    return chosen_action

def possible_actions(env):
    response = requests.post(API_ENDPOINT, json=input_data, headers=headers)
    # gpt(prompt_wrap(env.render()), model='gpt-4', n=1)[None]*3
    # TODO 这里还需要确定一下action的格式和调用方式 -- need verification n种第一句话
    actions = []
    for i in range(3):
        actions = [r.text for r in (env1.parse_response(response[i]))]
    return actions

def possible_actions(state, idx):
    obs = env1.reset(idx)
    response = [None] * 3
    for i in range(3):
        response[i] = openai.Completion.create(
            engine="text-davinci-003",
            prompt=state,
            max_tokens=1
    )

    possible_action = [r.text for r in response]
    return possible_action
# LLAMA or GPT-4

def main():
    #M_theta = GenerativeLLM()  
    #optimizer = optim.Adam(M_theta.parameters(), lr=0.001)
    initial_state = env1.prompt_wrap(env1.render())
    reward = 0.0
    N = 10  # Number of rollouts
    depth = 3  # Depth of exploration
    depth_limit = 5  # Depth limit
    iterations = 10  # Training iterations
    T = 5
     
# "prompt", reward, finish    env.
    for iteration in range(5, 11):  # Training iterations 5-10
        # Data Collection
        training_data = []

        for _ in range(T):
            rollouts = []
            Q = {}
            N_count = {}
            for _ in range(N):
                state = initial_state
                history = []
                generated_actions = set()
                

                for n in range(depth_limit):
                    if n < depth:
                        if state in generated_actions:
                        
                            pass
                        else:
                            response = requests.post(API_ENDPOINT, json=input_data, headers=headers)
                            # TODO Prompt wraping需要重新看一下，主要是json如何按顺序读入，应该需要一个循环
                             
                            possible_action = response.choices[0].text

                            # state 更新  不太确定
                            state = state + possible_action

                            generated_actions.add(state) # TODO 更新prompt 动态的算prompt
                            
                            N[state] = {}
                            Q[state] = {}
                            
                            N[state][possible_action] = 0
                            Q[state][possible_action] = 0
                            
                        # Select action using PUCT
                        chosen_action = PUCT(Q[state], N_count[state], n)
                            
                        # Expand by applying chosen action
                        next_state = state + chosen_action
                        if env1.answered(next_state):
                            break

                        history.append(state)
                        state = next_state
                    # else:
                    #     # best action
                    #     best_action = get_best_action(state)
                    #     history.append(state)
                    #     state += best_action

                # Backpropagation
                # 需要backpropagate到history中的每一个state ???
                reward = env1.reward(next_state)
                update_value(next_state, reward, chosen_action, Q, N_count)
                        

            # Finished one rollout
            a_star = []
            state = initial_state
            for n in range(depth_limit):
                # 判断environment是否finished
                if env1.answered(state):
                    break
                best_action = get_best_action(state)
                a_star.append(best_action)
                state += best_action
            # history.append(state)
            # rollouts.append(history)
            training_data.append(state)
        
        # Fine-tune LLM
        # def finetune(): 
        #     optimizer.zero_grad()
        #     loss = loss_function() 
        #     loss.backward()
        #     optimizer.step()

if __name__ == "__main__":
    main()



