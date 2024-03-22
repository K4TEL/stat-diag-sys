from diallama.mw_loader import Dataset, DataLoader
from diallama.database import MultiWOZDatabase
import json
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

dataset = Dataset(name, context_len=None)
data_loader = DataLoader(dataset, batch_size=5, shuffle=True, collate=False)
database = MultiWOZDatabase()
with open("results.txt", "w+") as f:
    print("=" * 20, "Database", "=" * 60)
    for time_str in ["noon", "1pm", "three forty five", "after 3pm", "lunch", "morning", "7:23", "ten o'clock"]:
        print(f"{time_str}:\t {database.time_str_to_minutes(time_str)}", file=f)
    print("=" * 100, file=f)
    for domain, query in [("restaurant", {"area": "center", "pricerange": "cheap"}),
                          ("train", {"arriveby": "12pm", "departure": "Cambridge", "day": "Tuesday"}),
                          ("hotel", {"pricerange": "cheap", "area": "west", "internet": "yes", "type": "guesthouse"})]:
        print(f"{(domain, query)}:\t {len(database.query(domain, query))}")
    print("=" * 20, "Data", "=" * 60)
    for idx, batch in enumerate(data_loader):
        print(f"Batch {idx} normal:", file=f)
        print(json.dumps(batch, indent=2), file=f)
        print(f"Number of tokens in batch {idx}:", file=f)
        print([sum([len(utt.split()) for utt in x['context']])+len(x['utterance'].split())+sum([len(slots) for slots in x['belief_state'].values()]) for x in batch], file=f)
        collate_batch = data_loader.collate_fn(batch)
        print(f"Batch {idx} collate:", file=f)
        print(collate_batch, file=f)
        print(f"Number of tokens in batch {idx} after collate:", file=f)
        print([len(cont)+len(utt)+len(bs) for cont, utt, bs in zip(collate_batch["context"], collate_batch["utterance"], collate_batch["belief_state"])], file=f)
