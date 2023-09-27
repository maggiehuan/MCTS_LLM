from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", load_in_8bit=True)

# class KeywordsStoppingCriteria(StoppingCriteria):
#     def __init__(self, keywords_ids: list):
#         self.keywords = keywords_ids

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         if input_ids[0][-1] in self.keywords:
#             return True
#         return False
    
        
# stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(keywords_ids = [13])])


# Test generate

prompt = "Hey, are you conscious? Can you talk to me?\n"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=100, eos_token_id=13, num_beams=5, num_return_sequences=5)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(generate_ids)
print(generate_ids.shape)

length_prompts = len(prompt)
outputs = [output[length_prompts:] for output in outputs]

print(outputs)

# Test generate one sentence