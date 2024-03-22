
import os
import json
from logzero import logger

DEADLINE = '2023-10-28'
FILES = ['hw1/README.md',
         ('hw1/analysis.(py|ipynb)', 2)]


def check(files):

    errors = 0

    for pattern, matches in files.items():
        if pattern.startswith('hw1/README'):
            if os.path.getsize(matches[0]) < 500:
                logger.warning(f'File {pattern} is too small (<500 bytes).')
                errors += 1

        elif pattern.startswith('hw1/analysis'):
            non_empty = [fname for fname in matches if os.path.getsize(fname) != 0]
            if not non_empty:
                logger.warning(f'No non-empty files for {pattern}.')
                errors += 1
                continue

            fname = non_empty[0]  # only checking the first non-empty
            with open(fname, 'r', encoding='UTF-8') as fh:
                data = fh.read()
            if fname.endswith('.ipynb'):
                data = json.loads(data)
                try:
                    srcs = ''
                    for cell in data['cells']:
                        for src in cell['source']:
                            srcs += src + '\n'
                    data = srcs
                except Exception as e:
                    logger.warning(f'Cannot parse {fname}: {str(e)}')
                    errors += 1
                    continue

            try:
                compile(data, fname, 'exec')
            except Exception as e:
                logger.warning(f'Cannot compile {fname}: {str(e)}')
                errors += 1

    return errors
