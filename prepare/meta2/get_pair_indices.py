# -*- coding:utf-8-*-
# By Jake

import os
import sys
import numpy as np
from scipy import sparse
import pandas as pd
import pickle
import gc


# Some global variables
g_load_path = '../../Data'
g_save_path = '../../Data/2015.07.04.primitive'
g_num_neg = 0


def load_as_dataframe(fn):
    """load csv file into memory"""
    df = pd.read_csv(fn)
    return df


def get_match_idx(df1, df2, kw='drawbridge_handle'):
    """matching"""
    matched_idx = []
    for id1, handle in enumerate(df1[kw]):
        id2 = np.where(df2[kw] == handle)[0].tolist()
        if len(id2) == 0:
            continue
        for x in id2:
            matched_idx.append( (id1, x) )
    return matched_idx


def get_unmatch_idx(df1, df2, num, kw='drawbridge_handle'):
    """unmatching"""
    unmatched_idx = []
    while len(unmatched_idx) < num:
        id1 = np.random.randint(len(df1[kw]))
        id2 = np.random.randint(len(df2[kw]))
        if df1[kw][id1] != df2[kw][id2]:
            unmatched_idx.append((id1, id2))
    return unmatched_idx


def main():
    dev_df = load_as_dataframe(os.path.join(g_load_path, 'dev_train_basic.csv'))
    coo_df = load_as_dataframe(os.path.join(g_load_path, 'cookie_all_basic.csv'))
    print('loading done')
    # start matching and unmatching
    matched_idx = get_match_idx(dev_df, coo_df)
    global g_num_neg
    if g_num_neg <= 0:
        g_num_neg = len(matched_idx)
    unmatched_idx = get_unmatch_idx(dev_df, coo_df, g_num_neg)
    print('matched and unmatched indices generated and saving to ' + os.path.join(g_save_path, 'indices.pkl'))
    # save both matched and unmatched indices
    indices = {'pos': matched_idx, 'neg': unmatched_idx}
    os.system('mkdir -p ' + g_save_path)
    with open(os.path.join(g_save_path, 'indices.pkl'), 'wb') as idf:
        pickle.dump(indices, idf)
        idf.close()
    print('saving done')
    

if __name__ == '__main__':
    main()
