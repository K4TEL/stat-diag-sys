# stat-diag-sys

### GPT-2 based NN connected to external knowledge base with 2-stage decoding for the Statistical Dialogue Systems course

Most of the code templates for training and evaluation of GPT-2 model were derived from the base repository of the course NPFL099

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


## Automated Tests

You can run some basic sanity checks for some homework assignments (so far, only the first one is imlemented,
we'll let you know if we create more).
Note that the tests require stuff from `requirements.txt` to be installed in your Python environment.
The tests assume checking in the current directory, they assume you have the correct branches set up.

For instance, to check `hw1`, run:

```
./run_tests.py hw1
```

By default, this will just check your local files. If you want to check whether you have
your branches set up correctly, use the `--check-git` parameter.
Note that this will run `git checkout hw1` and `git pull`, so be sure to save any 
local changes beforehand! 

Please always update from upstream before running tests.
Some may only be available at the last minute, we're sorry for that!
