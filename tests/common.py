
import subprocess
import datetime
from logzero import logger
import os
import re


def croak(message):
    """Print error message and fail."""
    logger.error(message)
    raise Exception(message)



def git_pull(path):
    """Runs git pull, crashes on fail."""
    logger.info('Running git pull...')
    output = subprocess.run(f"git pull", cwd=path, shell=True, check=True, capture_output=True)
    if output.returncode != 0:
        croak('Could not pull %s' % path)


def git_checkout_branch(hw_id, path, branch_name=None):
    """Checks out the desired git branch, crashes on fail."""
    if branch_name is None:
        branch_name = hw_id
    logger.info(f'Checking out {hw_id}...')
    output = subprocess.run(f"git checkout {branch_name} --", cwd=path, shell=True, capture_output=True)
    if output.returncode != 0:
        croak('Could not checkout branch %s for %s' % (hw_id, path))


def git_check_date(path, deadline):
    """Checks the date of the last commit in git (assuming the correct branch is already chosen), logs any problems found.
    Returns 1 on errors found, 0 otherwise."""
    logger.info('Checking last commit date...')
    output = subprocess.run("git show -s --format=%ct HEAD", cwd=path, shell=True, capture_output=True)
    if output.returncode != 0:
        croak('Could not get last commit date for %s' % path)

    commit_date = datetime.datetime.fromtimestamp(int(output.stdout.strip()))
    deadline_date = datetime.datetime.strptime(deadline + ' 23:59:59', '%Y-%m-%d %H:%M:%S')

    if (deadline_date - commit_date).days < 0:
        if (deadline_date - commit_date).days < - 14:
            logger.error('Last commit date (%s) is > 14 days past the deadline (%s)' % (commit_date.strftime('%Y-%m-%d %H:%M'), deadline))
        else:
            logger.warn('Last commit date (%s) is past the deadline (%s)' % (commit_date.strftime('%Y-%m-%d %H:%M'), deadline))
        return 1
    logger.info('Last commit date (%s) is before the deadline (%s)' % (commit_date.strftime('%Y-%m-%d %H:%M'), deadline))
    return 0


def check_file_presence(path, files):
    """Checks for the presence of given files under a path prefix. Accepts a list of singular paths or tuples
    (regex pattern, expected number of matches). Logs any problems found. Returns number of errors found."""
    logger.info('Checking for files...')

    errors = 0
    file_matches = {}
    for fname in files:
        # number of files to match a given pattern
        if isinstance(fname, tuple):
            fname, fnum = fname
            dirname, fpattern = os.path.split(fname)
            dirname = os.path.join(path, dirname)
            if not os.path.isdir(dirname):
                logger.warn("Found 0 files matching pattern `%s', expected %d" % (fname, fnum))
                errors += 1
                continue
            matches = [os.path.join(dirname, f) for f in os.listdir(dirname)
                       if os.path.isfile(os.path.join(dirname, f)) and re.match(fpattern, f)]
            if len(matches) != fnum:
                logger.warn("Found %d files matching pattern `%s', expected %d" % (len(matches), fname, fnum))
                errors += 1
            if matches:
                file_matches[fname] = matches
        else:
            if not os.path.isfile(os.path.join(path, fname)):
                logger.warn("Did not find file %s" % fname)
                errors += 1
            else:
                file_matches[fname] = [os.path.join(path, fname)]

    return errors, file_matches

def find_component(config, component_pattern):
    """Find component matching the given regex in the config."""
    return any([re.match(component_pattern,
                         next(iter(c.keys())) if isinstance(c, dict) else c)  # either dict with parameters, or plain string
                for c in config['components']])
