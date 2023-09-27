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

from accelerate import Accelerator
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import SFTTrainer
import torch
import time

class Hyperparams:
    depth_limit = 20
    rollout_num = 10  # Number of rollouts
    data_num_per_training = 10
    train_iterations = 5

model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
model_choice = "gpt-4"
API_KEY = '01c99186da2344b6a3f0f20748e08c73'
# API_KEY = os.environ.get("OPENAI_API_KEY")
API_ENDPOINT = f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"
headers = {'Content-Type': 'application/json', 'api-key': API_KEY}

def get_possible_actions_gpt(env: CrosswordsEnv, state):
    while True:
        response = requests.post(API_ENDPOINT, json=env.get_input_data(state), headers=headers)
        response_data = response.json()
        if 'error' in response_data:
            message = response_data['error']['message']
            print(message)
            sleep_time = re.findall(r'Please retry after (\w+) second', message)[0]
            time.sleep(int(sleep_time) + 1.0)
        else:
            break

    # with open('data.json', 'r') as f:
    #     response_data = json.load(f)

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
    while True:
        response = requests.post(API_ENDPOINT, 
                                json=env.get_input_data(state, num=1, stop_endline=False), 
                                headers=headers)
        response_data = response.json()
        if 'error' in response_data:
            message = response_data['error']['message']
            print(message)
            sleep_time = re.findall(r'Please retry after (\w+) second', message)[0]
            time.sleep(int(sleep_time) + 1.0)
        else:
            break
    
    # with open('data.json', 'w') as f:
    #     json.dump(response_data, f)

    generation = response_data['choices'][0]['message']['content']
    generation = generation.replace('\r\n', '\n')
    # print(generation)
    return generation

def selection(Q, N, state, action_list, c=1.0):
    # PUCT 
    total_visit_num = sum(N[state][a] for a in action_list)
    values = np.array(
        [Q[state][a] + c * np.sqrt(total_visit_num) / (N[state][a] + 1) for a in action_list]
    )
    chosen_action_idx = np.argmax(values)
    chosen_action = action_list[chosen_action_idx]
    return chosen_action

def update_value(Q, N, history_state_list, history_action_list, reward):
    for state, action in zip(history_state_list, history_action_list):
        Q[state][action] = (Q[state][action] * N[state][action] + reward) / (N[state][action] + 1)
        N[state][action] += 1

def mcts_construction(model, env: CrosswordsEnv, initial_state: str):
    # Construct a MCTS at state = initial_state
        
    Q, N, action_map = {}, {}, {}

    for rollout_idx in range(Hyperparams.rollout_num):
        # If there is only one possible action at current state, then stop searching. 
        if initial_state in action_map and len(action_map[initial_state]) == 1:
            break

        current_state = initial_state
        history_state_list, history_action_list = [], []

        # Selection process from the root node
        while current_state in action_map:
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
            possible_actions = get_possible_actions_gpt(env, current_state)
            
            # Add all the action into action_map, Q and N
            action_map[current_state] = possible_actions
            Q[current_state] = {}
            N[current_state] = {}
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
                print(final_state)
                reward = env.reward(final_state)
                print(f'get an answer with reward {reward}')
                
                # Update value
                update_value(Q, N, history_state_list + [current_state], 
                             history_action_list + [action], reward)
    
    # Finally, find the current best action. 
    best_action = selection(Q, N, initial_state, action_map[initial_state])
    return best_action

def rollout_once(model, env: CrosswordsEnv):
    current_state = ''

    for depth in range(Hyperparams.depth_limit):
        best_action = mcts_construction(model, env, current_state)
        current_state = current_state + best_action + '\n'
        if env.answered(current_state):
            break
    
    print('full generation:', current_state)
    
    return current_state

def cot_mcts(model, env: CrosswordsEnv, validation_env: CrosswordsEnv):
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
                generation = rollout_once(model, env)
            
            training_data.append((whole_prompt, generation))

        # TODO: Finetune Llama2 with training data
        pass

        # TODO: evaluate finetuned Llama2 on validation tasks
    
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gpu_id: Optional[int] = field(default=0)
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

def build_llama_model():
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        torch_dtype = torch.bfloat16
    else:
        quantization_config = None
        torch_dtype = None
    
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=script_args.gpu_id,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_auth_token=script_args.use_auth_token,
    )
    
    print('build model finished')
    return model

if __name__ == '__main__':
    # Load the model
    
    model = build_llama_model()
    
    # Build environment
    env = CrosswordsEnv()
    
    validation_env = CrosswordsEnv()
    
    cot_mcts(model, env, validation_env)
    
    if False:
        env_crosswords = CrosswordsEnv()
        env_crosswords.reset(0)
        state = """h1. An agendum; something to be done: TASKS\nh2. An engine: MOTOR\nh3. Pretentious; flowery: FANCY\nh4. A salon; a hall: PARLOR\nh5. To mock; to sneer: SCOFF\nv1. To heap: STACK\nv2. An Indian antelope: NILGA\nv3. To intend; to plan; to devise; a nettle; to guess: THINK\nv4. A nozzle: SPOUT\nv5. Desiccator; more dry: DRIER\nOutput:\nT A S K S\nM O T O R\nF A N C Y\nP A R L O R\nS C O F F"""
        print('answer:', env_crosswords.current_answer)
        reward = env_crosswords.reward(state)
        print(reward)
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
