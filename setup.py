#!/usr/bin/env python3

#    Copyright 2015 Roderick Bovee

import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# read in the version number

options = {
    'name': 'planknn',
    'version': '0.0.1',
    'description': 'A multilayer, convolutional neural network written in Python.',
    'author': 'Roderick Bovee',
    'author_email': 'rbovee@gmail.com',
    'license': 'MIT',
    'platforms': ['Any'],

    'classifiers': [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
    ],

    'packages': find_packages(),
    'install_requires': ['numpy', 'pandas', 'scikit-learn'],
    #TODO: also requires opencv2 -> not installable via pip

    'test_suite': 'nose.collector',
}
setup(**options)
