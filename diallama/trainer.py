from typing import Text, Optional, Tuple, List

import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import Optimizer, lr_scheduler
import torch.nn as nn
from logzero import logger

import re
import random
from fuzzywuzzy import fuzz

from diallama.mw_loader import DataLoader, SPECIAL_TOKENS, end_token
from diallama.database import MultiWOZDatabase

valid_scores_file = "../hw5/multiwoz_scores.txt"
test_outputs_file = "../hw5/multiwoz_outputs.txt"
test_f1_file = "../hw5/multiwoz_f1.txt"


class Trainer:
    def __init__(self,
                 model: PreTrainedModel,
                 train_data_loader: DataLoader,
                 valid_data_loader: DataLoader,
                 epochs: int,
                 optimizer: Optimizer,
                 scheduler: lr_scheduler._LRScheduler):
        if torch.cuda.is_available():
            self.model = model.to('cuda:0')
            self.device = model.device
        else:
            self.model = model
            self.device = "cpu"
        print(self.device)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        assert epochs > 0
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_criterion = nn.CrossEntropyLoss()

        # print(self.model.config.vocab_size)
        vocab = self.train_data_loader.tokenizer.get_vocab()
        self.model.resize_token_embeddings(len(vocab))
        # print(self.model.config.vocab_size)

    def train(self):
        self.model.train()
        train_batch_keys = ["input_ids", "attention_mask", "utterance_mask", "belief_mask"]
        logger.info('Starting training...')
        for epoch in range(self.epochs):
            logger.info(f'====== Epoch {epoch}/{self.epochs} Training ======')
            for step, batch in enumerate(tqdm.tqdm(self.train_data_loader)):
                """
                The batch is a dictionary of a form:
                {
                    "input_ids": Tensor[bs, maxlen],
                    "attention_mask": Tensor[bs, maxlen],
                    "belief_mask": Tensor[bs, maxlen],
                    "utterance_mask": Tensor[bs, maxlen],
                }
                """
                torch.cuda.empty_cache()

                batch = {k: v.to(self.device) for k, v in batch.items() if k in train_batch_keys}
                self.model.zero_grad()

                target_mask = torch.logical_or(batch["belief_mask"], batch["utterance_mask"])

                logits = self.model(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   use_cache=False,
                                   output_hidden_states=False,
                                   output_attentions=False).logits

                label_mask = torch.roll(target_mask, -1, 1)
                label_mask[:, -1] = 0
                labels = torch.roll(batch["input_ids"], -1, 1)
                labels[:, -1] = -100
                loss = self.loss_criterion(logits.view(-1, self.model.vocab_size)[target_mask.view(-1)],
                                           labels.view(-1)[label_mask.view(-1)])

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if step % 500 == 0:
                    for i, labels in enumerate(target_mask):
                        # logits = output.logits
                        pred = torch.argmax(logits[i, :], dim=1)
                        target = torch.argmax(logits[i, labels, :], dim=1)
                        true = batch["input_ids"][i, labels]
                        print("\nT Gen:\t", self.train_data_loader.tokenizer.decode(pred, skip_special_tokens=True))
                        print("\nT Pred:\t", self.train_data_loader.tokenizer.decode(target, skip_special_tokens=True))
                        print("T True:\t", self.train_data_loader.tokenizer.decode(true, skip_special_tokens=True))

                    # raise NotImplemented

            torch.cuda.empty_cache()
            logger.info(f'======= Epoch {epoch}/{self.epochs} Validation ===========')
            valid_loss, valid_token_acc_utter, valid_token_acc_bs, valid_perplexity = self.eval()
        logger.info('======= Final Validation ===========')
        final_loss, final_token_acc_utter, final_token_acc_bs, final_perplexity = self.eval()

    def eval(self, data_loader: Optional[DataLoader] = None) -> Tuple[float, float, float]:
        eval_batch_keys = ["input_ids", "attention_mask", "utterance_mask", "belief_mask"]
        self.model.eval()

        if data_loader is None:
            data_loader = self.valid_data_loader

        total_loss, total_perplexity = 0.0, 0.0
        total_tokens_utter, correct_tokens_utter = 0, 0
        total_tokens_bs, correct_tokens_bs = 0, 0

        with open(valid_scores_file, 'w') as file:
            file.truncate(0)

        with (torch.no_grad()):
            for step, batch in enumerate(tqdm.tqdm(data_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items() if k in eval_batch_keys}

                target_mask = torch.logical_or(batch["belief_mask"], batch["utterance_mask"])

                logits = self.model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    output_hidden_states=False,
                                    output_attentions=False).logits
                label_mask = torch.roll(target_mask, -1, 1)
                label_mask[:, -1] = 0
                labels = torch.roll(batch["input_ids"], -1, 1)
                labels[:, -1] = -100
                loss = self.loss_criterion(logits.view(-1, self.model.vocab_size)[target_mask.view(-1)],
                                           labels.view(-1)[label_mask.view(-1)])

                total_perplexity += torch.exp(loss)
                total_loss += loss.item()

                target_mask_utter = batch["utterance_mask"].view(-1)
                target_mask_bs = batch["belief_mask"].view(-1)

                predicted_tokens = logits.view(-1, logits.size(-1))
                true_tokens = batch["input_ids"].view(-1)

                correct_tokens_utter += (torch.argmax(predicted_tokens[target_mask_utter], dim=1) ==
                                   true_tokens[target_mask_utter]).sum().item()
                total_tokens_utter += target_mask_utter.sum().item()

                correct_tokens_bs += (torch.argmax(predicted_tokens[target_mask_bs], dim=1) ==
                                         true_tokens[target_mask_bs]).sum().item()
                total_tokens_bs += target_mask_bs.sum().item()

                # print(correct_tokens_utter, correct_tokens_bs, total_tokens_utter, total_tokens_bs)

                if step % 500 == 0:

                    for i, labels in enumerate(batch["belief_mask"]):
                        pred = torch.argmax(logits[i, :], dim=1)
                        select = torch.argmax(logits[i, labels, :], dim=1)
                        true = batch["input_ids"][i, labels]

                        print("\nE Gen:\t", self.train_data_loader.tokenizer.decode(pred, skip_special_tokens=True))
                        print("\nE BS Pred:\t", self.train_data_loader.tokenizer.decode(select, skip_special_tokens=True))
                        print("E BS True:\t", self.train_data_loader.tokenizer.decode(true, skip_special_tokens=True))
                        print(f"E BS Correct:\t{ (select == true).sum().item()} / {labels.sum().item()}")

                    for i, labels in enumerate(batch["utterance_mask"]):
                        pred = torch.argmax(logits[i, labels, :], dim=1)
                        true = batch["input_ids"][i, labels]

                        print("E Utter Pred:\t", self.train_data_loader.tokenizer.decode(pred, skip_special_tokens=True))
                        print("E Utter True:\t", self.train_data_loader.tokenizer.decode(true, skip_special_tokens=True))
                        print(f"E Utter Correct:\t{(pred == true).sum().item()} / {labels.sum().item()}")

                # accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
                # print(correct_tokens, total_tokens, accuracy)
                raise NotImplemented

            torch.cuda.empty_cache()

        valid_loss = total_loss / len(self.valid_data_loader)
        valid_perplexity = total_perplexity / len(self.valid_data_loader)
        valid_token_acc_utter = correct_tokens_utter / total_tokens_utter if total_tokens_utter > 0 else 0.0
        valid_token_acc_bs = correct_tokens_bs / total_tokens_bs if total_tokens_bs > 0 else 0.0

        logger.info(f'Loss: {valid_loss}')
        logger.info(f'Utter token accuracy: {valid_token_acc_utter}')
        logger.info(f'BS token accuracy: {valid_token_acc_bs}')
        logger.info(f'Perplexity: {valid_perplexity}')

        with open(valid_scores_file, 'a') as file:
            file.write(f"Utter token acc:\t{valid_token_acc_utter}\nBS token acc:\t{valid_token_acc_utter}\n"
                       f"Perplexity:\t{valid_perplexity}\nNLL Loss:\t{valid_loss}")

        return valid_loss, valid_token_acc_utter, valid_token_acc_bs, valid_perplexity


class GenerationWrapper:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 dataset: MultiWOZDatabase,
                 max_length: int = 30):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.db = dataset

        self.tokenizer.pad_token = "<|pad|>"
        self.model.config.pad_token_id = 50261
        self.model.generation_config.max_new_tokens = self.max_length
        self.model.generation_config.pad_token_id = 50261
        self.model.generation_config.bos_token_id = 50259
        self.model.generation_config.eos_token_id = 50258
        self.tokenizer.padding_side = "left"

    def interact(self):
        while True:
            ctx = []
            user_utterance = input('USER> ')
            user_utterance = user_utterance.strip()
            if user_utterance is None or len(user_utterance) == 0:
                print('Please, provide a nonempty utterance.')
                continue
            if user_utterance.lower() in ['stop', 'end', 'break']:
                break
            input = ' '.join(ctx[-4:]) + '<|endoftext|>' + user_utterance
            response = self.generate_single(user_utterance)
            print(f'SYSTEM> {response}')
            ctx.append('<|user|>' + user_utterance)
            ctx.append('<|system|>' + response)

    def _generate(self, prompts: List[Text]) -> List[Text]:
        self.model.eval()

        decoded = []
        for prompt in prompts:
            model_inputs = self.tokenizer(prompt,
                                          padding=True,
                                          return_tensors='pt',
                                          add_special_tokens=True).to(self.model.device)

            outputs = self.model.generate(**model_inputs,
                                          num_beams=5,
                                          no_repeat_ngram_size=2,
                                          top_k=50,
                                          top_p=0.95,
                                          temperature=1.0,
                                          do_sample=True,
                                          num_return_sequences=1)

            dec = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded.append(dec)

        return decoded

    def generate_single(self, prompt: Text) -> Text:
        model_input = f"<|user|> {prompt} <|belief|>"
        # print(model_input)
        decoded = self._generate([model_input])[0]

        pred_bs = decoded[len(prompt)+1:]
        bs_dict = parse_str_dict(pred_bs)

        db_res = {}
        for dom, query in bs_dict.items():
            db_res[dom] = len(self.db.query(dom, query))
        db_res = str(db_res).replace("'", " ")

        model_input = f"{model_input} {pred_bs} <|database|> {db_res} {end_token}"
        decoded = self._generate([model_input])[0]

        input_length = len(prompt)+len(db_res)+len(pred_bs)+5
        return decoded[input_length:]

    def generate_batch(self, prompts: List[Text]) -> List[Text]:
        batch_inputs = self.tokenizer(prompts,
                                      padding=True,
                                      return_tensors='pt',
                                      add_special_tokens=True).to(self.model.device)
        # print(batch_inputs)

        batch_outputs = self.model.generate(**batch_inputs,
                                            num_beams=5,
                                            no_repeat_ngram_size=2,
                                            top_k=50,
                                            top_p=0.95,
                                            temperature=1.0,
                                            do_sample=True,
                                            num_return_sequences=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in batch_outputs]
        return decoded_outputs

    def generate_loader(self, data_loader):
        with open(test_outputs_file, 'w') as file:
            file.truncate(0)

        self.model.eval()

        batch_num = len(data_loader)
        random_n = 100 if batch_num > 100 else 5
        random_samples = random.sample(range(0, batch_num), random_n)

        true_bs, pred_bs = [], []

        decoded_outputs = []
        with (torch.no_grad()):
            for b, batch in enumerate(data_loader):
                # print(len(true_bs), len(pred_bs))

                inputs = []
                for example in batch:
                    # print(example)
                    inputs.append(" ".join(example["context"]) + f" <|belief|>")

                    if b in random_samples:
                        true_bs.append(example["belief_state"])

                dec_out = self.generate_batch(inputs)

                next_inputs = []
                for i, out in enumerate(dec_out):
                    request = inputs[i]
                    for t in SPECIAL_TOKENS:
                        request = request.replace(t, "")

                    response = out[len(request):]
                    dict_bs = parse_str_dict(response)

                    if b in random_samples:
                        pred_bs.append(dict_bs)

                    db_res = {}
                    if len(dict_bs.keys()) != 0:
                        for domain, props in dict_bs.items():
                            db_res[domain] = len(self.db.query(domain, props))

                    db_str = str(db_res).replace("'", "")

                    next_inputs.append(f"{' '.join(example['context'])} <|belief|> {response} <|database|> {db_str}")

                final_out = self.generate_batch(next_inputs)

                for i, out in enumerate(final_out):
                    request = next_inputs[i]
                    for t in SPECIAL_TOKENS:
                        request = request.replace(t, "")

                    response = out[len(request):]

                    record = f"{response} <-- {request}"
                    print(record)
                    decoded_outputs.append(record)

                # raise NotImplemented

        slots_stats = f1_slots(true_bs, pred_bs)
        for k, v in slots_stats.items():
            print(k, v)

        with open(test_f1_file, 'w') as file:
            for k, v in slots_stats.items():
                file.write(f"{k}\t:\t{v['F1']}\n")

        log_outputs = decoded_outputs[:100]
        with open(test_outputs_file, 'a') as file:
            for out in log_outputs:
                file.write(out + "\n")


def parse_str_dict(input_string):
    # print(input_string)
    elements = re.split(r'\s*,\s*', input_string)

    result_dict = {}
    current_category = None

    for element in elements:
        # print(element)
        if '{' in element:
            recognized_bracs = re.findall(r'{(.*?){', input_string)
            if len(recognized_bracs) == 0:
                continue
            recognized_cats = re.findall(r'[a-zA-Z]+', recognized_bracs[0])
            if len(recognized_cats) == 0:
                continue

            current_category = recognized_cats[0]
            result_dict[current_category] = {}

            slot = element[element.rfind('{')+1:].split(":")
            key, value = slot[0].strip(), ":".join(slot[1:]).replace("}", "").strip()

            if key is None:
                continue
            else:
                result_dict[current_category][key] = value

        elif '}' in element:
            slot = element[:element.find('}')-1].split(":")
            key, value = slot[0].strip(), ":".join(slot[1:]).strip()

            if key is None:
                continue
            elif current_category in result_dict.keys():
                result_dict[current_category][key] = value
                current_category = None
        else:
            slot = element[:element.find('}') - 1].split(":")
            key, value = slot[0].strip(), ":".join(slot[1:]).strip()

            if key is None:
                continue
            elif current_category in result_dict.keys():
                result_dict[current_category][key] = value

    return result_dict


def f1_slots(true, pred):
    slots_stats = {}

    for i, true_bs in enumerate(true):
        pred_bs = pred[i]

        for t_dom, t_props in true_bs.items():
            for slot in t_props.keys():
                if slot not in slots_stats.keys():
                    slots_stats[slot] = {"TP": 0,
                                         "FP": 0,
                                         "FN": 0}

                slot_pred = False

                for p_dom, p_props in pred_bs.items():
                    for p_name, p_val in p_props.items():
                        if p_name == slot:
                            slot_pred = True

                            if fuzz.partial_ratio(p_val, t_props[slot]) > 0.5:
                                slots_stats[slot]["TP"] += 1
                            else:
                                slots_stats[slot]["FP"] += 1

                if not slot_pred:
                    slots_stats[slot]["FN"] += 1

    # for k, v in slots_stats.items():
    #     print(k, v)

    for slot, stats in slots_stats.items():
        if (stats["TP"] + stats["FN"]) > 0:
            stats["rec"] = stats["TP"] / (stats["TP"] + stats["FN"])
        else:
            stats["rec"] = 0

        if (stats["TP"] + stats["FP"]) > 0:
            stats["pre"] = stats["TP"] / (stats["TP"] + stats["FP"])
        else:
            stats["pre"] = 0

        # print(slot, stats)
        if (stats["pre"] + stats["rec"]) > 0:
            stats["F1"] = 2 * (stats["pre"] * stats["rec"]) / (stats["pre"] + stats["rec"])
        else:
            stats["F1"] = 0

    return slots_stats


# def memory_usage_stats(label=""):
#     def bytes_to_human_readable(size):
#         for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
#             if size < 1024.0:
#                 break
#             size /= 1024.0
#         return "{:.2f} {}".format(size, unit)
#
#     cpu_memory = psutil.virtual_memory()
#     print(f"{label} CPU\t {bytes_to_human_readable(cpu_memory.available)}: "
#           f"{bytes_to_human_readable(cpu_memory.used)} / "
#           f"{bytes_to_human_readable(cpu_memory.total)}")
#
#     if torch.cuda.is_available():
#         print(f"{label} GPU\t {bytes_to_human_readable(torch.cuda.memory_reserved())} / "
#               f"{bytes_to_human_readable(torch.cuda.memory_allocated())}")

    # for var in dir():
    #     print(var, type(eval(var)), eval(var), sys.getsizeof(eval(var)))

    # print(globals())
    # print(locals())


# def comp_loss(input_batch, model, criterion):
#     model.train()
#     train_batch_keys = ["input_ids", "labels", "attention_mask"]
#
#     batch = {k: v.to(model.device) for k, v in input_batch.items()} # if k in train_batch_keys}
#     model.zero_grad()
#
#     logits = model(input_ids=batch["input_ids"],
#                     attention_mask=batch["attention_mask"],
#                     use_cache=False,
#                     output_hidden_states=False,
#                     output_attentions=False).logits
#
#     loss = criterion(logits.view(-1, model.vocab_size), batch["labels"].view(-1))
#     # print(loss, loss.device)
#
#     loss.backward()
#
#     for k, v in batch.items():
#         batch[k] = v.detach().cpu() if model.device != "cpu" else v.detach()
#         batch[k].grad = None
#         batch[k].storage().resize_(0)
#
#     logits = logits.detach().cpu() if model.device != "cpu" else logits.detach()
#     logits.grad = None
#     logits.storage().resize_(0)
#
#     del logits, batch
#
#     return loss.item()
#
#
# def comp_metric(input_batch, model, criterion):
#     eval_batch_keys = ["input_ids", "labels", "attention_mask", "target_mask"]
#
#     batch = {k: v.to(model.device) for k, v in input_batch.items()} # if k in eval_batch_keys}
#
#     logits = model(input_ids=batch["input_ids"],
#                    attention_mask=batch["attention_mask"],
#                    use_cache=False,
#                    output_hidden_states=False,
#                    output_attentions=False).logits
#
#     loss = criterion(logits.view(-1, model.vocab_size),
#                                batch["labels"].view(-1))
#     perplexity = torch.exp(loss).item()
#
#     target_mask = batch["target_mask"].view(-1)
#
#     correct_tokens = (torch.argmax(logits.view(-1, logits.size(-1))[target_mask], dim=1) ==
#                        batch["labels"].view(-1)[target_mask]).sum().item()
#     total_tokens = target_mask.sum().item()
#     loss = loss.item()
#
#     for k, v in batch.items():
#         batch[k] = v.detach().cpu() if model.device != "cpu" else v.detach()
#         batch[k].grad = None
#         batch[k].storage().resize_(0)
#     target_mask = target_mask.detach().cpu() if model.device != "cpu" else target_mask.detach()
#     logits = logits.detach().cpu() if model.device != "cpu" else logits.detach()
#     logits.grad, target_mask.grad = None, None
#     logits.storage().resize_(0)
#     target_mask.storage().resize_(0)
#
#     del logits, batch, target_mask
#
#     return loss, perplexity, correct_tokens, total_tokens



