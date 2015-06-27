#-*- coding:utf-8 -*-
# By Jake

import sys
sys.path.append('..')
from modules import *

import numpy as np
import pandas as pd


def test_merge():
    """test merge function"""
    a = np.random.rand(4, 5)
    a = (a > 0.5) * a
    b = np.random.rand(4, 5)
    b = np.random.rand(8, 5)
    b = (b > 0.5) * b
    a = np.hstack((np.array([1, 2, 3, 4]).reshape(4, 1), a))
    b = np.hstack(( np.array(([1, 4, 5, 3, 6, 7, 8, 9])).reshape(8, 1), b ))
    c = pd.DataFrame(a).to_sparse(0)
    d = pd.DataFrame(b).to_sparse(0)
    print c
    print d
    print sparse_merge(c, d, left_on=0, right_on=0)


def main():
    test_merge()


if __name__ == '__main__':
    main()
