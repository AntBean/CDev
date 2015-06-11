#-*- coding: utf-8 -*-
"""
config file.
By Jake Zhao
"""

import os
from collections import defaultdict

config = defaultdict(defaultdict)

# Data
config['data']['trainpath'] = 'path/to/train'
config['data']['testpath'] = 'path/to/test'


# model
config['model'] = []
config['model'].append({'name': 'lr'})

# train
config['train'] = {}

# test
config['test']['flag'] = True

# main
config['main']['save'] = 'path/to/save'
