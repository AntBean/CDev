#-*- coding: utf-8 -*-
# By Jake

import os
import sys
import pickle
import subprocess

# global variables
g_index_dummy = {'dev': [1, 2, 3, 4, 5, 7, 8],
                 'coo': [1, 2, 3, 4, 5, 7]}
g_load_path = '../../DataSample'
g_save_path = '../../DataSample'


def get_uniq_column(fn, idx):
    '''opens a sub-thread to get uniq strings'''
    p = subprocess.Popen(['./get_uniq.sh', str(idx) + ' ' + fn],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out
