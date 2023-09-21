import numpy as np
import json
import sys
sys.path.append('../..')
from src.tot.prompts.crosswords import cot_prompt
from src.tot.models import gpt
from src.mcts.crossword_mcts import CrosswordsEnv

import random
import os
import requests
import json
import re
from collections import defaultdict
import pickle

class Hyperparams:
    depth_limit = 20
    rollout_num = 10  # Number of rollouts
    data_num_per_training = 10
    train_iterations = 5

model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
model_choice = "gpt-4-32k"
API_KEY = '01c99186da2344b6a3f0f20748e08c73'
# API_KEY = os.environ.get("OPENAI_API_KEY")
API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"
headers = {'Content-Type': 'application/json', 'api-key': API_KEY}

def get_possible_actions(env: CrosswordsEnv, state):
    response = requests.post(API_ENDPOINT, json=env.get_input_data(state), headers=headers)
    response_data = response.json()

    # with open('data.json', 'r') as f:
    #     response_data = json.load(f)

    print(response_data)
    actions = [choice['message']['content'].removesuffix('\r') for choice in response_data['choices']]
    
    actions = list(set(actions))
    return actions
    
    # # TODO 这里需要删掉所有的/r和/n, 然后把相同的句子分类 done
    # for i in range(6):
    #     content = response_data['choices'][i]['message']['content']
    #     content = content.split('\n')
    #     if content[0] not in unique_actions:
    #         #unique_actions.append(content)
    #         unique_actions = content + unique_actions
    #         num_actions += 1
    #     # print(content)
    # print(unique_actions)
    # return unique_actions, num_actions

def generate_full(env: CrosswordsEnv, state):
    response = requests.post(API_ENDPOINT, 
                             json=env.get_input_data(state, num=1, stop_endline=False), 
                             headers=headers)
    response_data = response.json()

    generation = response_data['choices'][0]['message']['content']
    generation = generation.replace('\r\n', '\n')
    # print(generation)
    return generation

def selection(Q, N, state, action_list, c=1.0):
    # PUCT 
    total_visit_num = sum(N[state][a] for a in enumerate(action_list))
    values = np.array(
        [Q[state][a] + c * np.sqrt(total_visit_num) / (N[state][a] + 1) for a in action_list]
    )
    chosen_action_idx = np.argmax(values)
    chosen_action = action_list[chosen_action_idx]
    return chosen_action_idx, chosen_action

def update_value(Q, N, history_state_list, history_action_list, reward):
    for state, action in zip(history_state_list, history_action_list):
        Q[state][action] = (Q[state][action] * N[state][action] + reward) / (N[state][action] + 1)
        N[state][action] += 1

def mcts_construction(env: CrosswordsEnv, initial_state):
    # Construct a MCTS at state = initial_state
    
    Q, N, action_map = {}, {}, {}

    for rollout_idx in range(Hyperparams.rollout_num):
        # If there is only one possible action at current state, then stop searching. 
        if initial_state in action_map and len(action_map[initial_state]) == 1:
            break

        current_state = initial_state
        history_state_list, history_action_list = [], []

        # Selection process from the root node
        while action_map in current_state:
            best_action = selection(Q, N, current_state, action_map[current_state])
            history_state_list.append(current_state)
            history_action_list.append(best_action)
            current_state = current_state + best_action + '\n'
            # if generate success, finish
            # TODO: check env.answered implementation
            if env.answered(current_state):
                break
        
        if env.answered(current_state):
            # Finished generation from MCTS. 
            # Then, we will compute reward, and update value, and do nothing else. 
            reward = env.reward(current_state)
            update_value(Q, N, history_state_list, history_action_list, reward)
        else:
            # Meet an unfinished node in MCTS. 
            # We will first expand this node (generate all possible actions), 
            #    and generate the full result from each of the action, 
            #    and compute reward and update value. 
            # Optionally, we can add all the generation into MCTS, but the implementation will 
            #    take more time. 
            
            # Generate possible actions
            possible_actions = get_possible_actions(env, current_state)
            
            # Add all the action into action_map, Q and N
            action_map[current_state] = possible_actions
            for action in possible_actions:
                Q[current_state][action] = 0.0
                N[current_state][action] = 0
            
            # Generate until finish for each action, and compute the reward
            for action in possible_actions:
                next_state = current_state + action + '\n'
                if env.answered(next_state):
                    final_state = next_state
                else:
                    generation = generate_full(env, next_state)
                    final_state = next_state + generation + '\n'
                reward = env.reward(final_state)

                # Update value
                update_value(Q, N, history_action_list + [current_state], 
                             history_action_list + [action], reward)
    
    # Finally, find the current best action. 
    best_action = selection(Q, N, initial_state, action_map[initial_state])
    return best_action

def rollout_once(env: CrosswordsEnv):
    current_state = ''

    for depth in range(Hyperparams.depth_limit):
        best_action = mcts_construction(env, current_state)
        current_state = current_state + best_action + '\n'
        if env.answered(current_state):
            break
    
    return current_state

def mcts_cot(env: CrosswordsEnv, validation_env: CrosswordsEnv):
    # The main generation and training loop for one environment. 
    # TODO: implement full evaluation on all tasks

    # TODO: optional behavior cloning training for Llama2. 
    
    for iteration in range(Hyperparams.train_iterations):
        training_data = []
        for dataset_idx in range(Hyperparams.data_num_per_training):
            # Reset environment to random task
            env.reset_random()
            whole_prompt = env.get_whole_prompt()
            for i in range(Hyperparams.data_num_per_training):
                generation = rollout_once()
            
            training_data.append((whole_prompt, generation))

        # TODO: Finetune Llama2 with training data
        pass

        # TODO: evaluate finetuned Llama2 on validation tasks
    


if __name__ == '__main__':
    env_crosswords = CrosswordsEnv()
    env_crosswords.reset(0)
    generate_full(env_crosswords, '')
    # possible_actions(None, '')
    if False:
        prompt_path = "../tot/data/crosswords/mini0505.json"

        env_crosswords = CrosswordsEnv(prompt_path)
        env_crosswords.reset(0)
        input_data = env_crosswords.get_input_data('')

        # print(input_data['messages'][1]['content'])
        # input('.')

        response = requests.post(API_ENDPOINT, json=input_data, headers=headers)

        # response_data = response.json()

        # with open('data.json', 'w') as f:
        #     json.dump(response_data, f)

        # print(response_data)
