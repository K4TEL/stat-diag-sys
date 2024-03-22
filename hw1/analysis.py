import ijson
import re
import math
from collections import Counter
import statistics
import sys

file = 'train.json'
small_file = "test_random_split.json"

system = ["0_Wizard", "1_Wizard"]
user = ["0_Apprentice", "1_Apprentice"]

user_size = {'dialogs': 0, 'turns': 0, 'sentences': 0, 'words': 0}
sys_size = {'dialogs': 0, 'turns': 0, 'sentences': 0, 'words': 0}

sys_vocab, user_vocab = [], []

user_counts = {'turns': [], 'sentences': [], 'words': []}
sys_counts = {'turns': [], 'sentences': [], 'words': []}


def shannon_entropy(word_list):
    word_count = len(word_list)
    word_freq = Counter(word_list)

    entropy = 0.0
    for word in word_freq:
        probability = word_freq[word] / word_count
        entropy -= probability * math.log2(probability)

    return entropy


with open(file, 'rb') as file:
    parser = ijson.parse(file)

    diag_start = False
    wizard = False

    diag_user_size = {'turns': 0, 'sentences': 0, 'words': 0}
    diag_sys_size = {'turns': 0, 'sentences': 0, 'words': 0}

    for prefix, event, value in parser:
        if event == 'string':
            key = prefix.split(".")[-1]

            if key == "speaker":
                if diag_start:
                    if value == system[0]:
                        sys_size["dialogs"] += 1
                    else:
                        user_size["dialogs"] += 1

                    diag_start = False

                if value in system:
                    sys_size["turns"] += 1
                    diag_sys_size["turns"] += 1
                else:
                    user_size["turns"] += 1
                    diag_user_size["turns"] += 1

                wizard = value in system

            if key == "text":
                sentences_count = len(value.split("."))
                words = re.sub(r'[^\w\s]', ' ', value).split()
                words_count = len(words)

                if wizard:
                    sys_size["sentences"] += sentences_count
                    sys_size["words"] += words_count
                    sys_vocab += words

                    diag_sys_size["sentences"] += sentences_count
                    diag_sys_size["words"] += words_count

                else:
                    user_size["sentences"] += sentences_count
                    user_size["words"] += words_count
                    user_vocab += words

                    diag_user_size["sentences"] += sentences_count
                    diag_user_size["words"] += words_count

                print(f"{'SYS' if wizard else 'USER'}: {value}")

            if key == "chosen_topic":
                diag_start = True

                sys_counts["words"].append(diag_sys_size["words"])
                sys_counts["sentences"].append(diag_sys_size["sentences"])
                sys_counts["turns"].append(diag_sys_size["turns"])

                user_counts["words"].append(diag_user_size["words"])
                user_counts["sentences"].append(diag_user_size["sentences"])
                user_counts["turns"].append(diag_user_size["turns"])

                for key in diag_sys_size.keys():
                    diag_sys_size[key] = 0

                for key in diag_user_size.keys():
                    diag_user_size[key] = 0

                print(f"\tDialog topic:\t{value}")


sys_size["vocabulary"], user_size["vocabulary"] = len(set(sys_vocab)), len(set(user_vocab))
sys_size["shannon_entropy"], user_size["shannon_entropy"] = shannon_entropy(sys_vocab), shannon_entropy(user_vocab)

for key, val in {key: statistics.mean(sys_counts[key]) for key in sys_counts}.items():
    sys_size[f"mean_{key}"] = val

for key, val in {key: statistics.stdev(sys_counts[key]) for key in sys_counts}.items():
    sys_size[f"stddev_{key}"] = val

for key, val in {key: statistics.mean(user_counts[key]) for key in user_counts}.items():
    user_size[f"mean_{key}"] = val

for key, val in {key: statistics.stdev(user_counts[key]) for key in user_counts}.items():
    user_size[f"stddev_{key}"] = val

# output_file = open("report.txt", "w")
# sys.stdout = output_file

print("\nUSER\tstatistics:")
for key, val in user_size.items():
    print(f"{key}:\t{val}")

print("\nSYS\tstatistics:")
for key, val in sys_size.items():
    print(f"{key}:\t{val}")

# sys.stdout = sys.__stdout__
# output_file.close()