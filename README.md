# stat-diag-sys

### GPT-2 based NN connected to external knowledge base with 2-stage decoding

Most of the code templates for training and evaluation of GPT-2 model were derived from the existing base repository (private)

Business logic of training, data preprocessing and evaluation was implemented manually

## Covered tasks:

- **Dataset Exploration**
- **MultiWOZ 2.2 Loader**
- **Finetuning GPT-2 on MultiWOZ**
- **MultiWOZ 2.2 DB + State**
- **Two-stage decoding**

## Installation

The code here requires Python 3 and [pip](https://pypi.org/project/pip/).

For a basic installation, clone the repository and run:
```
cd <your-repo>; pip install [--user] -r requirements.txt
```

However, you probably also want to use the repo as a set of libraries (e.g. import packages from it).
This will be used in some of the test scripts for the individual assignments. Unless you always run
these scripts from your repository's base directory, you can do a full in-place install of 
your cloned repository:
```
cd <your-repo>; pip install [--user] -e .
```

Use `--user` to install into your user directories (recommended unless you're using 
a [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://docs.conda.io/en/latest/)).