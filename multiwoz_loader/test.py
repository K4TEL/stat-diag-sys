from diallama.mw_loader import Dataset, DataLoader
import json
import os
import sys
import random
import torch
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

name = "test"
if len(sys.argv) > 1:
    name = sys.argv[1]
dataset = Dataset("train", context_len=None)
data_loader = DataLoader(dataset, batch_size=5, shuffle=True, collate=False)
with open(os.path.join(os.path.dirname(__file__), f"results_{name}.txt"), "w+") as f:
    for idx, batch in enumerate(data_loader):
        print(f"Batch {idx} normal:", file=f)
        print(json.dumps(batch, indent=2), file=f)
        collate_batch = data_loader.collate_fn(batch)
        print(f"Batch {idx} collate:", file=f)
        print(collate_batch, file=f)
        print(f"Number of tokens in batch {idx}:", file=f)
        print([sum([len(utt.split()) for utt in x['context']])+len(x['utterance'].split()) for x in batch], file=f)
        print(f"Number of tokens in batch {idx} after collate:", file=f)
        print([len(cont)+len(utt) for cont, utt in zip(collate_batch["context"], collate_batch["utterance"])], file=f)
        break

    dataset = Dataset("train")
    print("Train", len(dataset), file=f)
    dataset = Dataset("validation")
    print("Valid", len(dataset), file=f)
    dataset = Dataset("test")
    print("Test", len(dataset), file=f)
