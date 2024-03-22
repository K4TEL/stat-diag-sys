from transformers import GPT2LMHeadModel

from train import MODEL_PATH, CTX_LEN, BS
from diallama.mw_loader import Dataset, DataLoader
from diallama.trainer import GenerationWrapper

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
train_dataset, valid_dataset = Dataset('train', context_len=CTX_LEN), Dataset('validation', context_len=CTX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, collate=True)
valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=True, collate=True)
gen_wrapper = GenerationWrapper(model, train_loader.tokenizer, max_length=40)
print(gen_wrapper.generate_single('I am looking for a restaurant in the centre that serves british food.'))
print(gen_wrapper.generate_single('I want some cheap hotel to stay for 4 nights.'))
print(gen_wrapper.generate_single('Can you recommend any attractions?'))

test_dataset = Dataset('test', context_len=CTX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True, collate=False)
gen_wrapper.generate_loader(test_loader)


