import pickle
import os
import copy

import datasets
import torch
import torch.utils.data as torchdata
from torchtext.data.iterator import BucketIterator as TorchBucketIterator
import transformers

from diallama.database import MultiWOZDatabase

end_token = "<|endoftext|>"
pad_token = "<|pad|>"
SPECIAL_TOKENS = ["<|user|>", "<|system|>", "<|belief|>", "<|database|> ", pad_token, end_token]


class Dataset(torchdata.Dataset):
    """
    Dataset class, inherits from torch.utils.data.Dataset.
    Load the MultiWoz dataset using huggingface.datasets
    Able to shorten the context length by setting context_len.
    """
    def __init__(self, split, context_len=None, cache_dir="./"):
        self.split = split
        self.fields = {}
        # Create cache dir if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if torch.cuda.is_available():
            self.f_name = os.path.join(cache_dir, f"{split}_preprocessed_data.json")
        else:
            self.f_name = os.path.join(cache_dir, f"{split}_preprocessed_data_small.json")
        self.database = MultiWOZDatabase()
        # If the dataset has already been preprocessed, load it from the cache
        if os.path.isfile(self.f_name):
            data = pickle.load(open(self.f_name, 'rb'))
            print(f"Loaded {len(data)} examples from cached file.")
        else:
            dataset = datasets.load_dataset(path='multi_woz_v22', split=split, streaming=True)
            data = []
            cnt = 0
            for idx, dialogue in enumerate(dataset):
                cnt += 1
                if cnt > 3:
                    break
                if idx % 500 == 0:
                    print(f"Processing dialogue {idx + 1}")
                data.extend(self.parse_dialogue_into_examples(dialogue, context_len=context_len))
            self.save_data(data)
        self.data = data

    def save_data(self, data):
        assert not os.path.exists(self.f_name), f"{self.f_name} already exists."
        with open(self.f_name, 'wb+') as f:
            pickle.dump(data, f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def parse_dialogue_into_examples(self, dialogue, context_len=None):
        """
        Parses a dialogue into a list of examples.
        Each is a dictionary of the folowing structure:
        {
            # for HW2:
            'context': list[str],  # list of utterances preceeding the current utterance
            'utterance': str,  # the string with the current response
            'delex_utterance': str,  # the string with the current response which is delexicalized, i.e. slot values are
                                    # replaced by corresponding slot names in the text.
            # for HW3:
            'belief_state': dict[str, dict[str, str]],  # belief state dictionary, for each domain a separate belief state dictionary,
                                                        # choose a single slot value if more than one option is available
            'database_results': dict[str, int] # dictionary containing the number of matching results per domain
        }
        The context can be truncated to k last utterances.


        Existing services:
            {'hotel', 'restaurant', 'police', 'bus', 'train', 'attraction', 'hospital', 'taxi'}
        Existing intents:
            {'find_bus', 'find_train', 'find_restaurant', 'find_attraction', 'book_hotel', 'find_taxi',
            'find_police', 'book_train', 'find_hotel', 'find_hospital', 'book_restaurant'}
        Existing slots_values_names:
            {'bus-departure', 'hotel-pricerange', 'train-departure', 'hotel-bookstay', 'hotel-bookday',
            'restaurant-bookpeople', 'restaurant-booktime', 'restaurant-pricerange', 'attraction-type',
            'restaurant-name', 'bus-destination', 'train-bookpeople', 'hotel-area', 'taxi-departure',
            'taxi-destination', 'attraction-area', 'attraction-name', 'restaurant-area', 'taxi-arriveby',
            'hotel-stars', 'restaurant-bookday', 'taxi-leaveat', 'hotel-bookpeople', 'restaurant-food',
            'train-destination', 'hospital-department', 'hotel-parking', 'hotel-type', 'train-leaveat',
            'bus-leaveat', 'train-day', 'hotel-name', 'hotel-internet', 'train-arriveby', 'bus-day'}
        """
        examples = []
        turns = dialogue['turns']

        utterance, speaker, diag_acts, frame = turns["utterance"], turns["speaker"], turns["dialogue_acts"], turns["frames"]

        special_utterance = []
        for i, utter in enumerate(utterance):
            special_utterance.append(f"{SPECIAL_TOKENS[speaker[i]]} {utter}")

        for i in range(speaker.index(1), len(speaker), 2):
            k = i - context_len if context_len is not None and i > context_len else 0

            bs, db_res = {}, {}

            frames = frame[k:i]
            for f in frames:
                if len(f["service"]) == 0:
                    continue

                domain = f["service"][0]
                slot_names = f["state"][0]["slots_values"]["slots_values_name"]
                slot_vals = f["state"][0]["slots_values"]["slots_values_list"]

                dom_props = {}
                for j, name in enumerate(slot_names):
                    dom_props[name.split("-")[-1]] = slot_vals[j][0]

                res = self.database.query(domain, dom_props)

                bs[domain] = dom_props
                db_res[domain] = len(res)

            span_info = diag_acts[i]["span_info"]
            if len(span_info["act_slot_name"]) > 0:
                span_start, span_end = span_info["span_start"], span_info["span_end"]
                span_name = span_info["act_slot_name"]

                utter = list(utterance[i])
                for start, end, name in zip(reversed(span_start), reversed(span_end), reversed(span_name)):
                    utter[start:end] = [f"[{name}]"]

                delex_uter = ''.join(utter)
            else:
                delex_uter = utterance[i]

            example = {
                'context': special_utterance[k:i],
                'utterance': utterance[i],
                'delex_utterance': delex_uter,
                'belief_state': bs,
                'database_results': db_res
            }
            examples.append(example)

        return examples


class MyBucketIterator(TorchBucketIterator):
    """
    BucketIterator from torchtext.data, overriding the __iter__ method to yield a simple
    batch without PyTorch's Fields.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        """
        Copied from torchtext.data.BucketIterator, but with the following changes:
        `yield minibatch` instead of `yield Batch(minibatch, self.dataset, self.device)`
        """
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield minibatch
            if not self.repeat:
                return


class DataLoader:
    """
    Iteratively returns batches of batch_size from given dataset, optionally shuffled. If collate=True, returns
    integer tokens instead of strings using huggingface.transformers.GPT2Tokenizer.

    Inside a batch, each example has a similar number of tokens, both when tokenizing and when not. To achieve this,
    the sort function is different each time. Slightly edited pytorchtext.legacy.data.BucketIterator is used for bucketing
    the batches and sampling from the batches.
    """

    def __init__(self, dataset, batch_size, shuffle=True, collate=False):
        def _sort_examples_to_buckets_f(example):
            # print(example)
            full_example = f'{" ".join(example["context"])} {example["utterance"]} {example["belief_state"]} {example["database_results"]}'
            # print(full_example)
            if self.collate:
                return len(self.tokenizer.encode(full_example, add_special_tokens=True))
            else:
                return len(full_example.split())

        self.dataset = dataset
        self.batch_size = batch_size
        self.collate = collate
        # BucketIterator
        self.iterator = MyBucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            sort_key=_sort_examples_to_buckets_f,
            shuffle=shuffle,
            sort_within_batch=True
        )
        self.iterator.create_batches()
        # Tokenizer with special tokens
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2",)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS})

    def __iter__(self):
        # Apply collate_fn if desired
        collate = self.collate_fn if self.collate else lambda x: x
        try:
            for batch in self.iterator:
                yield collate(batch)
        except StopIteration:
            self.iterator.create_batches()

    def __len__(self):
        return len(self.iterator)

    def tokenize(self, sentence):
        """
        Uses pretrained GPT2Tokenizer from huggingface.transformers to tokenize a sentence.
        """
        return self.tokenizer(sentence, add_special_tokens=True)["input_ids"]

    def collate_fn(self, batch):
        """
        Use transformers.GPT2Tokenizer to convert the batch to a single dictionary (output) of the following structure:
        output = {
        # for HW2:
        'context': list[list[int]],     # tokenized context (list of subword ids from all preceding dialogue turns,
                                        # system turns prepended with `<|system|>` token and user turns with `<|user|>`)
                                        # for all batch examples
        'utterance': list[list[int]],   # tokenized utterances (list of subword ids from the current dialogue turn)
                                        # for all batch examples
        'delex_utterance': list[list[int]], # tokenized and delexicalized utterances (list of subword ids
                                            # from the current dialogue turn) for all batch examples
        # for HW4 add:
        'belief_state': list[list[int]],    # belief state dictionary serialized into a string representation and prepended with
                                            # the `<|belief|>` special token and tokenized (list of subword ids
                                            # from the current dialogue turn) for all batch examples
        'database_results': list[list[int]],    # database result counts serialized into string prepended with the `<|database|>`
                                                # special token and tokenized (list of subword ids from the current dialogue turn)
                                                # for all batch examples
        }
        # for HW3:
        output = {
                "input_ids": Tensor[bs, maxlen], # concatenated ids for context and utterance,
                                                 # interleaved with the special tokens
                "attention_mask": Tensor[bs, maxlen], # mask, 1 for valid input, 0 for padding
                "context_mask": Tensor[bs, maxlen], # mask, 1 for context tokens, 0 for others
                "utterance_mask": Tensor[bs, maxlen], # mask, 1 for utterance tokens, 0 for others
        }
        """
        spec_token_ids = self.tokenize(SPECIAL_TOKENS)
        delim_token, space_token, bs_token, db_token = spec_token_ids[-1][0], spec_token_ids[-2][0], spec_token_ids[2][0], spec_token_ids[3][0]

        # print(space_token, self.tokenizer.padding_side, bs_token, db_token, delim_token, self.tokenizer.vocab_size)

        def masks_append(masks, key=""):
            for mask_key in masks.keys():
                if mask_key != key:
                    masks[mask_key].append(0)
                else:
                    masks[mask_key].append(1)

            return masks

        output = {"input_ids": [],
                  "attention_mask": [],
                  "context_mask": [],
                  "utterance_mask": [],
                  "belief_mask": [],
                  "database_mask": []}

        batch_tmp = copy.deepcopy(batch)

        for e in batch_tmp:
            e["belief_state"] = "<|belief|> " + str(e["belief_state"]).replace("'", " ")
            e["database_results"] = "<|database|> " + str(e["database_results"]).replace("'", " ")
            e["context"] = " ".join(e["context"])
            e["input"] = f'{e["context"]} {e["belief_state"]} {e["database_results"]} {end_token} {e["delex_utterance"]} {end_token}'

        for example in batch_tmp:
            example_tokenized = self.tokenizer(example['input'], add_special_tokens=True)

            output["input_ids"].append(example_tokenized["input_ids"])
            output["attention_mask"].append(example_tokenized["attention_mask"])

            masks = {
                "context": [],
                "utterance": [],
                "belief": [],
                "database": []
            }

            context, bs, db = True, True, True

            for id in example_tokenized["input_ids"]:
                if id == bs_token and context:
                    context = False
                    masks_append(masks, "context")

                elif id == db_token and bs:
                    bs = False
                    masks_append(masks, "belief")

                elif id == delim_token and db:
                    db = False
                    masks_append(masks, "database")

                elif id == delim_token and not db:
                    masks_append(masks, "utterance")

                else:
                    if context:
                        masks_append(masks, "context")
                    elif bs:
                        masks_append(masks, "belief")
                    elif db:
                        masks_append(masks, "database")
                    else:
                        masks_append(masks, "utterance")

            output["context_mask"].append(masks["context"])
            output["utterance_mask"].append(masks["utterance"])
            output["belief_mask"].append(masks["belief"])
            output["database_mask"].append(masks["database"])

        max_len = max(len(seq) for seq in output["input_ids"])
        for key in output.keys():
            # print(key, len(output[key][0]), output[key])
            type = torch.long if key == "input_ids" else torch.bool
            pad = space_token if key == "input_ids" else 0
            output[key] = torch.tensor([seq + [pad] * (max_len - len(seq))
                                        for seq in output[key]], dtype=type)

        # output["target_mask"] = torch.logical_or(output["belief_mask"], output["utterance_mask"])

        # output["labels"] = output["input_ids"] * output["target_mask"]
        # output["labels"][output["target_mask"] != 1] = -100

        # if output["input_ids"].shape[1] > 200:
        #     for e in batch_tmp:
        #         print(e["input"])

        # for key in output.keys():
        #     for i, ex in enumerate(output[key]):
        #         if key != "input_ids":
        #             print(key, ex.sum().item())
        #             print(self.tokenizer.decode(output["input_ids"][i, ex], skip_special_tokens=False))
        #         else:
        #             print(key, len(ex))
        # raise NotImplemented

        return output
