import re
import os
import json
from src.tot.tasks.base import Task, DATA_PATH
from src.tot.prompts.crosswords import * 
from src.tot.models import gpt
# import openai
import torch as th
# import torch.optim as optim
import numpy as np
from src.tot.prompts.crosswords import cot_prompt, chat_cot_prompt_small
import copy
#from tot.models import gpt

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask)


class GSM8KEnv:
    def __init__(self, file="/home/ziyu/code/dataset/grade-school-math/grade_school_math", reward_type='reward_rule'):
        # DATA_PATH = '/home/ziyu/code/LLMs/mcts-llm/src/tot/data/crosswords/mini0505.json'
        # self.file = os.path.join(DATA_PATH, 'crosswords', file)
        self.file = file
        self.file = json.load(open(self.file))
        self.task_num = len(self.file)
        self.reward_type = reward_type
        self.env_name = 'crosswords'

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
            if i < 5:
                section += f"h{i + 1}. {item}\n"
            else:
                section += f"v{i - 4}. {item}\n"
        section += "Thoughts:"
        return section

    def __len__(self): 
        return self.task_num

    def reward(self, output_raw: str):
        output = output_raw.split('Output:\n')[-1]
        letters = []
        for i, line in enumerate(output.strip().split('\n')[-5:], 1):
            letters_line = line.split(' ')[:5]
            letters_line += [' '] * (5 - len(letters_line))
            letters.extend(letters_line)

        letters = letters + [' '] * (25 - len(letters))
        
        if len(letters) != 25 or len(self.current_answer) != 25:
            print('error here')
            from IPython import embed; embed()
        
        reward_letter = 0
        reward_word = 0

        for i in range(0, len(letters), 5):
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
        if "Output:\n" in output:
            if len(output.strip().split('Output:\n')[-1].split('\n')) == 5:
                return True
        else:
            return False
        
    
    def get_ans(self, board):
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans

    def render_clues(self, status=None):
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + '\n'

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