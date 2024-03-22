import os.path
import random

import torch
import numpy as np
from transformers import AdamW, GPT2LMHeadModel, GPT2Config

from diallama.mw_loader import Dataset, DataLoader, SPECIAL_TOKENS
from diallama.trainer import Trainer

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

CTX_LEN = 4
BS = 12
EPOCHS = 5
MODEL_PATH = os.path.join(os.curdir, 'gpt2-multiwoz')

if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
    config = GPT2Config.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, config=config)
else:
    print("empty")
    train_dataset, valid_dataset = Dataset('train', context_len=CTX_LEN), Dataset('validation', context_len=CTX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, collate=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=True, collate=True)

    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(config)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    def linear_lr_lambda(batch_step):
        return max(0.0, 1.0 - batch_step / (len(train_loader) * EPOCHS))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lr_lambda)

    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        EPOCHS,
        optimizer,
        scheduler
    )
    trainer.train()
    model.save_pretrained(MODEL_PATH)