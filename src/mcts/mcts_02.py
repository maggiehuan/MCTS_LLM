import openai
import torch
import torch.optim as optim
import numpy as np
import json
from tot.prompts.crosswords import propose_prompt, value_prompt
from tot.models import gpt
from tot.tasks.crosswords import MiniCrosswordsEnv
import MiniCrosswordsEnv as env
import re
import copy
from tot.models import gpt

env = MiniCrosswordsEnv()

openai.api_key = "YOUR_API_KEY"

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
        reward = computed_reward(s_new)  
        update_value(s, reward)
    else:
        reward = rollout(s_new, depth - 1)

    # Backpropagation
    update_value(s, reward)
    return reward


def computed_reward(state):

    return torch.rand(1)  

def update_value(state, reward):
    Q[(state, a_best)] = (Q[(state, a_best)] * N_count[(state, a_best)] + reward) / (N_count[(state, a_best)] + 1)
    N_count[(state, a_best)] += 1

# 需要根据GPT-3的输出来判断是否已经回答完毕 需要设计一下prompt， 然后state的定义需要定义清楚
# 都变成environment的一部分， ENV.answer
def answered(state):
    if state == env.answered:
        return True
    else:
        return False

# prompt也是Env
history = env.prompt
a_star = []

# Main loop 
# for _ in range(N):
#     # 
#     s_0 = "prompt" 


#     for n in range(L):
#         if s_0 in Q:
#             a_best = max(range(num_actions), key=lambda a: Q[(s_0, a)] + C * np.sqrt(np.log(N_count[s_0]) / N_count[(s_0, a)]))
#         else:
#             a_best = np.random.choice(num_actions)
#             Q[(s_0, a_best)] = 0
#             N_count[(s_0, a_best)] = 0

#         N_count[s_0] += 1
#         a_star.append(a_best)
#         s_0 = s_0 + a_best
#         history.append(s_0)


def prompt_wrap(obs):
    return propose_prompt.format(input=obs)

def get_best_action(self):
    best_child = max(self.children, key=lambda node: node.visits)
    return self.get_action_to_reach_child(best_child)


def PUCT(Q, N, t, c=1.0):
    # PUCT 
    values = np.array([Q[a] + c * np.sqrt(np.log(t) / (N[a] + 1)) for a in Q])
    chosen_action = np.argmax(values)
    return chosen_action

def possible_actions(state):
    response = [None] * 3
    for i in range(3):
        response[i] = openai.Completion.create(
            engine="text-davinci-003",  
            prompt=state,
            max_tokens=1
    )
    # 这里possible action
    possible_action[] = response.choices[0].text
    return possible_action[]


def main():
    M_theta = GenerativeLLM()  
    optimizer = optim.Adam(M_theta.parameters(), lr=0.001)
    initial_state = "prompt"
    reward = 0.0
    N = 10  # Number of rollouts
    depth = 3  # Depth of exploration
    depth_limit = 5  # Depth limit
    
# "prompt", reward, finish    env.

    for iteration in range(5, 11):  # Training iterations 5-10
        # Data Collection
        training_data = []
        for _ in range(T):
            rollouts = []
            for _ in range(N):
                state = initial_state
                history = []
                generated_actions = set()
                
                for n in range(depth_limit):
                    if n < depth:
                        if state in generated_actions:
                        
                            pass
                        else:
                            response = openai.Completion.create(
                                engine="text-davinci-003",  
                                prompt=state,
                                max_tokens=1
                            )
                            # 生成n个action，封装成一个函数，每个action都需要更新一次searching，
                            possible_action = response.choices[0].text

                            # state 没有更新

                            generated_actions.add(state)
                            
                            N[state] = {}
                            Q[state] = {}
                            
                            N[state][possible_action] = 0
                            Q[state][possible_action] = 0
                            
                        # Select action using PUCT
                        chosen_action = PUCT(Q[state], N[state], n)
                            
                        # Expand by applying chosen action
                        next_state = state + chosen_action
                        if answered(next_state):
                            break
                        
                        # Backpropagation, reward computation, and value update
                        reward = computed_reward(next_state)
                        update_value(next_state, reward)
                        
                        # Update Q and N
                        N[state][chosen_action] += 1
                        Q[state][chosen_action] += reward
                        
                        history.append(state)
                        state = next_state
                    else:
                        # best action
                        best_action = get_best_action(state)
                        history.append(state)
                        state += best_action
                # Finished one rollout
                a_star = []
                state = initial_state
                for n in range(depth_limit):
                    best_action = get_best_action(state)
                    a_star.append(best_action)
                    state += best_action
                history.append(state)
                rollouts.append(history)
            training_data.extend(rollouts)
        
        # Fine-tune LLM
        optimizer.zero_grad()
        loss = loss_function() 
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()

