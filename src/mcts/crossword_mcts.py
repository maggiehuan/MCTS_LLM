import re
import os
import json
from src.tot.tasks.base import Task, DATA_PATH
from src.tot.prompts.crosswords import * 
from src.tot.models import gpt
# import openai
# import torch
# import torch.optim as optim
import numpy as np
from src.tot.prompts.crosswords import cot_prompt, chat_cot_prompt_small
import copy
#from tot.models import gpt

class CrosswordsEnv:
    def __init__(self, file="../tot/data/crosswords/mini0505.json", reward_type='reward_rule'):
        # DATA_PATH = '/home/ziyu/code/LLMs/mcts-llm/src/tot/data/crosswords/mini0505.json'
        # self.file = os.path.join(DATA_PATH, 'crosswords', file)
        self.file = file
        self.file = json.load(open(self.file))
        self.task_num = len(self.file)
        self.reward_type = reward_type

        self.task_inputs = [data[0] for data in self.file]
        self.task_answers = [data[1] for data in self.file]

    def reset_random(self):
        self.task_id = np.random.randint(0, self.task_num)
        self.current_input = self.task_inputs[self.task_id]
        self.current_answer = self.task_answers[self.task_id]

    def reset(self, task_id): 
        self.task_id = task_id 
        self.current_input = self.task_inputs[self.task_id]
        self.current_answer = self.task_answers[self.task_id]

    def get_input_data(self, state, num=5, stop_endline=True):
        # old_prompt = "Input: h1. A lunar valley\nh2. A fatty oil\nh3. To entice\nh4. To lower; to reduce\nh5. A solitary person\nv1. According to the roster\nv2. Another name for Port-Francqui\nv3. An illicit lover; a European lake\nv4. To lisp\nv5. To come in\nThoughts:\nh1. Presented; revealed: SHOWN\nh2. An interjection expressing sorrow: WIRRA\nh3. Benefit; result: AVAIL\nh4. A cigarette: RETTE\nh5. Chased up a tree: TREED\nv1. Swarthy; tawny: SWART\nv2. An apiarist or bee keeper: HIVER\nv3. To speak formally: ORATE\nv4. To indite; to scribble: WRITE\nv5. An insecticide: NALED\nOutput:R I L L E\nO L E I N\nT E M P T\nA B A S E\nL O N E R\nInput: {input}\n{state}"
        input_data = {  
            "messages": [  
                {"role": "system", "content": "You are a clever AI Assistant which carefully follow the instruction to solve the problem."},  
                {"role": "user", "content": self.get_whole_prompt(state=state)},
                #{"role": "user", "content": "Output:"}
            ],  
            "max_tokens": 500,
            "temperature": 0.7,
            "n":num,
            "stop": '\n' if stop_endline else []
        }

        return input_data
    
    def get_whole_prompt(self, state=''):
        input = self.get_input()
        return chat_cot_prompt_small.format(input=input, state=state)

    def get_input(self):
        section = ""
        for i, item in enumerate(self.current_input):
            #section += f"h{i + 1}. {{}}".format(item)
            if i < 5:
                section += f"h{i + 1}. {item}\n"
            else:
                section += f"v{i - 4}. {item}\n"
            #section += cot_prompt_format.format(input=section)
        section += "Thoughts:"
        #print(section)
        return section

    def __len__(self): # question
        return self.task_num

    def reward(self, output: str):
        output = output.split('Output:\n')[-1]
        letters = []
        for i, line in enumerate(output.strip().split('\n')[-5:], 1):
            letters_line = line.split(' ')[:5]
            letters_line += [' '] * (5 - len(letters_line))
            letters.extend(letters_line)
        reward_letter = 0
        reward_word = 0
        print(letters)

        for i in range(0, len(letters), 5):
            # print(letters[i:i+5])
            if letters[i:i+5] == self.current_answer[i:i+5]:
                reward_word += 1
        for i in range(5):
            if letters[i:25:5] == self.current_answer[i:25:5]:
                reward_word += 1
        reward_word = reward_word / 10

        for i in range(25):
            if letters[i] == self.current_answer[i]:
                reward_letter += 1
        reward_letter = reward_letter / 25

        reward_map = {'reward_letter': reward_letter, 'reward_word': reward_word, 'reward_rule': reward_letter}
        return reward_map[self.reward_type]

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