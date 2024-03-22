#!/usr/bin/env python3

from setuptools import setup
from setuptools import find_packages

setup(
    name='diallama',
    version='0.0.1dev',
    description='Framework for a dialogue system using pretrained language models',
    author='ÚFAL Dialogue Systems Group, Charles University',
    author_email='odusek@ufal.mff.cuni.cz',
    url='https://gitlab.com/ufal/dsg/diallama',
    download_url='https://gitlab.com/ufal/dsg/diallama.git',
    license='Apache 2.0',
    include_package_data=True,
    install_requires=['logzero',
                      'python-Levenshtein',
                      'fuzzywuzzy',
                      'datasets>=2.7.1',
                      'torch',
                      'torchtext==0.6.0',
                      'transformers',
                      'numpy'],
    packages=find_packages()
)

