import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.crosswords import * 
from tot.models import gpt
# import openai
# import torch
# import torch.optim as optim
import numpy as np
from tot.prompts.crosswords import cot_prompt
import copy
#from tot.models import gpt

class CrosswordsEnv:

    def __init__(self, file):
        DATA_PATH = '/home/ziyu/code/LLMs/mcts-llm/src/tot/data/crosswords/mini0505.json'
        self.file = os.path.join(DATA_PATH, 'crosswords', file)
        self.file = json.load(open(self.file))
        self.n = len(self.file)
        self.idx = 0
        self.data = None
        self.board_gt = None
        self.times = 0
        self.status = [0] * 10

        self.input_data = self.file[0][0]
        #print(self.input_data)
        self.processed_input = self.prompt_input()

    def get_input_data(self, state):
        input = self.prompt_input()

        input_data = {  
            "messages": [  
                {"role": "system", "content": "Solve 5x5 mini crosswords. Given an input of 5 horizontal clues and 5 vertical clues, generate thoughts about which 5-letter word fits each clue, then an output of 5 rows, where each row is 5 letter separated by space."},  
                {"role": "user", "content": "Input: h1. A lunar valley\nh2. A fatty oil\nh3. To entice\nh4. To lower; to reduce\nh5. A solitary person\n\
v1. According to the roster\nv2. Another name for Port-Francqui\nv3. An illicit lover; a European lake\nv4. To lisp\nv5. To come in\n\
Thoughts:\nh1. Presented; revealed: SHOWN\nh2. An interjection expressing sorrow: WIRRA\nh3. Benefit; result: AVAIL\nh4. A cigarette: RETTE\nh5. Chased up a tree: TREED\n\
v1. Swarthy; tawny: SWART\nv2. An apiarist or bee keeper: HIVER\nv3. To speak formally: ORATE\nv4. To indite; to scribble: WRITE\nv5. An insecticide: NALED\n\
Output:R I L L E\nO L E I N\nT E M P T\nA B A S E\nL O N E R\n\
Input: {input}\n{state}".format(input=input, state=state)},
                #{"role": "user", "content": "Output:"}
            ],  
            "max_tokens": 500,
            "temperature": 0.7,
            "n":5
        }

        # append一下state

        return input_data

    def prompt_input(self):
        section = ""
        for i, item in enumerate(self.input_data):
            #section += f"h{i + 1}. {{}}".format(item)
            if i < 5:
                section += f"h{i + 1}. {item}: \n"
            else:
                section += f"v{i - 4}. {item}: \n"
            #section += cot_prompt_format.format(input=section)
        section += "Thoughts:"
        #print(section)
        return section

    def __len__(self): # question
        return self.n

    def prompt(self):
        print(cot_prompt_format.format(input=self.input_data))
        return cot_prompt.format(input=self.input_data) 
        

    def reset(self, idx): 
        self.idx = idx 
        self.data = self.file[idx][1] 
        self.board_gt = self.file[idx] 
        print(self.data)

    def reward(self, output: str):
        output = output.split('Output:\n')[-1]
        letters = []
        for i, line in enumerate(output.strip().split('\n')[-5:], 1):
            letters.extend(line.split(' ')[:5])
        reward_w = 0
        reward_letter = 0
        print(letters)

        for i in range(0, len(letters), 5):
            # print(letters[i:i+5])
            if letters[i:i+5] == self.data[i:i+5]:
                reward_letter += 1
        reward_letter = reward_letter / 5

        for i in range(25):
            if letters[i] == self.data[i]:
                reward_w += 1
        reward = reward_w / 25

        return reward, reward_letter

    def answered(self, output: str):
        if output == "Output:\n":
            if len(output.split('Output:\n')[-1].split('\n')) == 5:
                return True
        else:
            return False
        # response = gpt(self.prompt_wrap(env.render()), model='gpt-4', n=1)[0]
        # if response == "answered":
        #     return True
        # else:
        #     return False

    def get_ans(self, board):
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans

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
        self.env = CrosswordsEnv(file)  # use it as a stateless tool
        self.xs = []
        for idx in range(len(self.env)):
            self.env.reset(idx)
            self.xs.append(self.env.render_clues())
        self.steps = 10  
        self.cache_proposals = {}

    def __len__(self) -> int:
        return len(self.env)


    def set_status(self, x: str, y: str):
        idx = self.xs.index(x)
        self.test_output(idx, y) 

    def get_input(self, idx: int) -> str:
        self.env.reset(idx)
        return self.env.render_clues()


### 测试实例
if __name__ == '__main__':
    file='/home/ziyu/code/LLMs/mcts-llm/src/tot/data/crosswords/mini0505.json'
    env = CrosswordsEnv(file)
    env.reset(5)
    onput = '''R I L L E
O L E I N
T E M P T
A B A S E
L O N E R'''
    # output = [   "A",
    #         "G",
    #         "E",
    #         "N",
    #         "D",
    #         "M",
    #         "O",
    #         "T",
    #         "O",
    #         "R",
    #         "A",
    #         "R",
    #         "T",
    #         "S",
    #         "Y",
    #         "S",
    #         "A",
    #         "L",
    #         "L",
    #         "E",
    #         "S",
    #         "L",
    #         "E",
    #         "E",
    #         "R"]
    env.reward(onput)