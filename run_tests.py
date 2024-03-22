#!/usr/bin/env python3

from argparse import ArgumentParser
import sys
import importlib
from logzero import logger
import os

from tests.common import git_checkout_branch, git_pull, git_check_date, check_file_presence


def main(hw_id, path=None, check_git=False):

    if not path:
        # Using directory of this file
        path = os.path.dirname(os.path.realpath(__file__))
    logger.info(f'Checking {hw_id} in {path}...')

    # checking path validity (will crash on fail)
    if check_git:
        try:
            git_checkout_branch(hw_id, path, 'master')
        except Exception:
            git_checkout_branch(hw_id, path, 'main')
        git_pull(path)
        git_checkout_branch(hw_id, path)
        git_pull(path)

    sys.path.insert(0, path)

    test_module = importlib.import_module(f"tests.{hw_id}")

    # checking files and dates
    errors = 0
    if check_git:
        errors += git_check_date(path, test_module.DEADLINE)
    file_errs, file_matches = check_file_presence(path, test_module.FILES)
    errors += file_errs

    # running actual checks
    errors += test_module.check(file_matches)

    # report total number of errors/issues
    logger.info('Found %d issues.' % errors)


if __name__ == '__main__':

    ap = ArgumentParser(description='Running lab homework assignment tests.')

    ap.add_argument('--check-git', help='Try changing branch + git pull', action='store_true')
    ap.add_argument('--use-path', help='Don\'t check this repo itself, but a specified path', type=str)
    ap.add_argument('hw_id', help='Assignment ID (hw1, hw2, etc.)',
                    choices=['hw1'])

    args = ap.parse_args()
    main(args.hw_id, path=args.use_path, check_git=args.check_git)
