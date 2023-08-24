import re
import os
import json
from src.tot.tasks.base import Task, DATA_PATH
from src.tot.prompts.crosswords import * 
from src.tot.models import gpt
# import openai
# import torch
import torch.optim as optim
import numpy as np
from src.tot.prompts.crosswords import propose_prompt, value_prompt
import copy
from src.tot.models import gpt


# prompt, get prompt, answered, reward, update value, rollout, computed reward,
# class ENV:


# ##读取文件 说明和answer的要求
# ##组合prompt 用说明组合prompt
# 用answer组合reward
# 用下面的env组合我们自己的env-> 先看懂每个函数，然后哪些是我们需要的or可以直接用的
# 没有的需要设计一下
# pdb调试工具的学习，一些基本功能
# 实现env，然后把mcts02的部分换成env
# 完善MCTS file


class CrosswordsEnv:

    def prompt_wrap(observation):
            return propose_prompt.format(input=observation) 
            breakpoint()


    def parse_line(input_str):
        # regular expression pattern to match the input string format
        pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'

        # use regex to extract the parts of the input string
        match = re.match(pattern, input_str)

        if match:
            # extract the matched groups
            parts = [match.group(1), match.group(2), match.group(3)]
            return parts
        else:
            return None

    confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  

    def parse_response(response):
        # split the response into lines
        lines = response.split('\n')

        # parse each line
        parsed_lines = [parsed_lines(line) for line in lines]

        # filter out the lines that didn't match the format
        parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), confidence_to_value.get(line[2], 0)) for line in parsed_lines if line is not None]

        return parsed_lines if len(parsed_lines) >= 1 else None


    #def get_candidates_to_scores(env): ## Reward
    def reward(env):
        obs = env.render()
        if obs in env.cache: 
            print('cache hit')
            return env.cache[obs]
        print('call gpt')
        responses = gpt(prompt_wrap(obs), model='gpt-4', n=8)
        candidates_to_scores = {}
        for response in responses:
            parsed_response = parse_response(response)
            if parsed_response:
                for candidate, score in parsed_response:
                    candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
            # choose candiate with highest score
        # print(sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True))
        env.cache[obs] = candidates_to_scores
        return candidates_to_scores

    # def propose_score(env, idx): # computed reward
    def computed_reward(env, idx):
        obs = env.reset(idx)
        done = False
        infos = []
        while not done:
            responses = gpt(prompt_wrap(obs), model='gpt-4', n=5)
            candidates_to_scores = {}
            for response in responses:
                parsed_response = parsed_response(response)
                if parsed_response:
                    for candidate, score in parsed_response:
                        candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
            # choose candiate with highest score
            print(sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True))
            if len(candidates_to_scores) == 0:
                break
            candidates =  sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True)
            for candidate in candidates:
                env_ = copy.deepcopy(env)
                env_.step(candidate)
                if not any(_ == 2 for _ in env_.status):
                    break
            print(candidate)
            # candidate = input()
            obs, r, done, info = env.step(candidate)
            print(obs)
            print(env.steps, info)
            print('-------------------\n\n\n')
            infos.append(info)
        return infos
    
    # def update_value(state, reward, a_best, Q, N_count):
    #     Q[(state, a_best)] = (Q[(state, a_best)] * N_count[(state, a_best)] + reward) / (N_count[(state, a_best)] + 1)
    #     N_count[(state, a_best)] += 1
    #     return Q, N_count

    def update_value(env, idx, propose_score, Q, N_count):
        # updated value should be the average of all the propose scores
        Q[idx] = (Q[idx] * N_count[idx] + propose_score) / (N_count[idx] + 1)
        N_count[idx] += 1
        return Q, N_count

        # obs = env.reset(idx)
        # done = False
        # infos = []
        # while not done:
        #     responses = gpt(prompt_wrap(obs), model='gpt-4', n=5)
        #     candidates_to_scores = {}
        #     for response in responses:
        #         parsed_response = parsed_response(response)
        #         if parsed_response:
        #             for candidate, score in parsed_response:
        #                 candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
        #     # choose candiate with highest score
        #     print(sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True))
        #     if len(candidates_to_scores) == 0:
        #         break
        #     candidates =  sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True)
        #     for candidate in candidates:
        #         env_ = copy.deepcopy(env)
        #         env_.step(candidate)
        #         if not any(_ == 2 for _ in env_.status):
        #             break
        #     print(candidate)
        #     # candidate = input()
        #     obs, r, done, info = env.step(candidate)
        #     print(obs)
        #     print(env.steps, info)
        #     print('-------------------\n\n\n')
        #     infos.append(info)
        # return infos
    
    def answered(env):
        response = gpt(prompt_wrap(env.render()), model='gpt-4', n=1)[0]
        if response == "answered":
            return True
        else:
            return False

    def rollout(env, s, depth, Q, N_count, C, num_actions):
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
        
        update_value(s, reward)
        return reward

    def possible_actions(env):
        response = gpt(prompt_wrap(env.render()), model='gpt-4', n=1)[None]*3
        actions = []
        for i in range(3):
            actions = [r.text for r in (parse_response(response[i]))]
        return actions
    

class MiniCrosswordsEnv: 
    def __init__(self, file='mini0505.json'): 
        self.file = os.path.join(DATA_PATH, 'crosswords', file) 
        self.file = json.load(open(self.file)) 
        self.n = len(self.file) 
        self.cache = {} 
        self.idx = None 
        self.times = 0 
        self.prompt_status_cache = {} 

    def __len__(self): 
        return self.n 
    
    def reset(self, idx, board=None, status=None, steps=None): 
        self.idx = idx 
        # random.sample(idx)
        # if class
        self.data, self.board_gt = self.file[idx] 
        self.board = ['_'] * 25 
        self.ans = ['_____'] * 10 
        breakpoint()
        self.ans_gt = self.get_ans(self.board_gt) 
        self.steps = 0 
        self.status = [0] * 10  # 0: unfilled; 1: filled; 2: filled then changed
        if board is not None: 
            self.board = board
            self.ans = self.get_ans(self.board) 
        if status is not None:
            self.status = status
        if steps is not None:
            self.steps = steps
        return self.render()
    

    def prompt_status(self):
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        for ans, data, status in zip(self.ans, self.data, self.status):
            # if status != 0: continue
            if ans.count('_') >= 4: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            prompt = value_prompt.format(input=line)
            if prompt in self.prompt_status_cache:
                res = self.prompt_status_cache[prompt]
            else:
                res = gpt(prompt)[0]
                self.prompt_status_cache[prompt] = res
            # print(line)
            # print(res)
            # print()
            res = res.split('\n')[-1].strip()
            if res in count: count[res] += 1
        # print(count)
        return count
    
    def render_gt_board(self):
        s = "GT Board:\n"
        for i in range(5):
            s += ' '.join(self.board_gt[i*5:(i+1)*5]) + '\n'
        return s
    
    def render_board(self):
        s = "Current Board:\n"
        for i in range(5):
            s += ''.join(self.board[i*5:(i+1)*5]) + '\n'
        return s

    def render_clues(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + '\n'
        return s
    
    def render_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + ': ' + self.ans[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + ': ' + self.ans[i] + '\n'
        return s
    
    def render_gt_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + ': ' + self.ans_gt[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + ': ' + self.ans_gt[i] + '\n'
        return s

    def render(self, status=True):
        if status:
            return self.render_board() + '\nUnfilled:\n' + self.render_ans(status=0) + '\nFilled:\n' + self.render_ans(status=1) + '\nChanged:\n' + self.render_ans(status=2)
        else:
            return self.render_board() + '\n' + self.render_ans()
    
    def get_ans(self, board):
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans
    
    def step(self, action):
        self.steps += 1
        action = action.split('\n')[-1]
        action = action.split('. ')
        if len(action) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}
        pos, word = action

        if len(word) != 5:
            return 'Invalid! Word should have 5 letters.', 0, False, {}
        if pos.startswith('h'):
            idx = int(pos[1:]) - 1
            self.board[idx*5:(idx+1)*5] = list(word.upper())
        elif pos.startswith('v'):
            idx = int(pos[1:]) - 1
            self.board[idx::5] = list(word.upper())
            idx += 5  # for later status update
        else:
            return 'Invalid! Position should be h1-h5 or v1-v5', 0, False, {}
        
        self.new_ans = self.get_ans(self.board)
        # self.status = [2 if (status == 1 and ans != new_ans) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status = [2 if any(letter != new_letter and letter != '_' for letter, new_letter in zip(ans, new_ans)) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status[idx] = 1
        self.ans = self.new_ans
        r_all = (self.board == self.board_gt)
        r_letter = sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10
        return self.render(), r_all, (r_all or self.steps >= 20), {'r_letter': r_letter, 'r_word': r_word, 'r_game': r_all}


class MiniCrosswordsTask(Task):
    """
    Input (x)   : Decription of a 5x5 mini crossword
    Output (y)  : List of 10 words to fill in the crossword
    Reward (r)  : word level and game level
    Input Example: 
    Output Example: 
    """
    def __init__(self, file):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        self.env = MiniCrosswordsEnv(file)  # use it as a stateless tool
        self.xs = []
        for idx in range(len(self.env)):
            self.env.reset(idx)
            self.xs.append(self.env.render_clues())
        self.steps = 10  # TODO: variable steps??
        self.cache_proposals = {}

    def __len__(self) -> int:
        return len(self.env)
    
    def get_input(self, idx: int) -> str:
        self.env.reset(idx)
        return self.env.render_clues()
    
    # def test_output(self, idx: int, output: str):  # TODO: r_word for now
    #     self.env.reset(idx)
    #     info = {'r_word': 0}
    #     for line in output.split('\n'):
    #         if line.startswith('h') or line.startswith('v'):
    #             _, _, _, info = self.env.step(line)
    #     return info['r_word']
    
    def test_output(self, idx: int, output: str):
        self.env.reset(idx)
        output = output.split('Output:\n')[-1]
        info = {'r_word': 0, 'r_letter': 0, 'r_game': 0}
        for i, line in enumerate(output.strip().split('\n')[-5:], 1):
            letters = line.split(' ')[:5]
            word = ''.join(letters)
            word = word + '_' * (5 - len(word))
            action = f'h{i}. {word}'
            # print(action)
            _, _, _, info = self.env.step(action)
        info['r'] = info['r_word']
        return info

    def set_status(self, x: str, y: str):
        idx = self.xs.index(x)
        self.test_output(idx, y)  # update self.env
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    def propose_prompt_wrap(self, x: str, y: str='') -> str:
        self.set_status(x, y)
        return propose_prompt.format(input=self.env.render())
    
    def propose_outputs_unwrap(self, x: str, y: str, outputs: list, n_max_propose: int) -> list:
        confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  # TODO: ad hoc
        proposals_to_scores = {}
        for output in outputs:
            lines = output.split('\n')
            pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'
            for line in lines:
                match = re.match(pattern, line)
                if match:
                    parts = [match.group(1), match.group(2), match.group(3)]
                    proposal = parts[0].lower() + '. ' + parts[1].lower()
                    score = confidence_to_value.get(parts[2], 0)
                    proposals_to_scores[proposal] = proposals_to_scores.get(proposal, 0) + score
        
        proposals = sorted(proposals_to_scores.items(), key=lambda x: x[1], reverse=True)
        if n_max_propose != -1:
            proposals = proposals[:n_max_propose]
        proposals = [y + proposal[0] + '\n' for proposal in proposals]
        self.cache_proposals[(x, y, n_max_propose)] = proposals
        return proposals
    
    def evaluate(self, x: str, y: str, n_evaluate_sample: int) -> int:
        self.set_status(x, y)
        assert n_evaluate_sample == 1 # TODO: ad hoc
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        for ans, data, status in zip(self.env.ans, self.env.data, self.env.status):
            if ans.count('_') >= 4: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            prompt = value_prompt.format(input=line)
            res = gpt(prompt)[0]
            print(line)
            print(res)
            print()
            res = res.split('\n')[-1].strip()
            if res in count: count[res] += 1
        print(count)
        return count

### 测试实例
if __name__ == '__main__':
    env = MiniCrosswordsEnv()
    env.reset(5)
