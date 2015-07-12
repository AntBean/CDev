from data import *

import os
import sys
import numpy as np
from scipy import sparse
from time import time


def testData():
    g_index_dummy = {'dev': [2, 3, 4, 5, 7, 8],
                     'coo': [2, 3, 4, 5, 7, 8]}
    g_index_drop = {'dev': [2], 'coo': [2]}
    g_load_path = '../Data'
    d = DataSource(g_index_dummy, g_index_drop, g_load_path)
    l = d.getNextBatch(32)
    print(l)
    # start testing time
    start = time()
    for i in range(10):
        l = d.getNextBatch(32)
    end = time()
    print('Time used for getting batch: {}'.format(end-start))



if __name__ == '__main__':
    testData()
