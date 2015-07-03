# -*- coding:utf-8-*-
# By Jake

# TODO
# header

import os
import numpy as np
from scipy import sparse
import pandas as pd
import pickle


# Some global variables
g_index_dummy = {'dev': [2, 3, 4, 5, 7, 8],
                 'coo': [2, 3, 4, 5, 7]}
g_load_path = '../../DataSample'
g_save_path = '../../DataSample/2015.07.02.primitive'
g_num_neg = 0


def load_as_dataframe(fn):
    """load csv file into memory"""
    df = pd.read_csv(fn)
    return df


def add_dummies(ds, idx, ref):
    """convert a data series to an array with dummy variables"""

    def to_dummies(string, ref):
        """find the position of string in ref, and derive dummies"""
        assert isinstance(ref, list)
        idx = ref.index(string)  # string must be in the ref
        arr = np.zeros(len(ref), )
        arr[idx] = 1
        return arr

    assert len(ref) == len(ds)  # references have to be padded
    assert isinstance(ds, pd.core.series.Series)
    assert isinstance(idx, list)
    arr = ds.values.copy()
    for idx_elem in idx:
        raw = arr[idx_elem]
        arr = np.delete(arr, idx_elem)
        np.insert(arr, idx_elem, to_dummies(raw, ref[idx_elem]))
    return arr


def padding(idx, ref, pad=[]):
    """padding the references"""
    assert isinstance(ref, list)
    assert isinstance(idx, list)
    for i, _ in enumerate(ref.keys()):
        if i not in idx:
            ref[i] = pad


def get_match_idx(df1, df2, kw='drawbridge_handle'):
    """matching"""
    matched_idx = []
    for id1, handle in enumerate(df1[kw]):
        id2 = np.where(df2[kw])[0].tolist()
        matched_idx.append(map(lambda x: (id1, x), id2))
    return matched_idx


def get_unmatch_idx(df1, df2, num, kw='drawbridge_handle'):
    """unmatching"""
    unmatched_idx = []
    while len(unmatched_idx) <= num:
        id1 = np.random.randint(len(df1[kw]))
        id2 = np.random.randint(len(df2[kw]))
        if df1[kw][id1] != df2[kw][id2]:
            unmatched_idx.append((id1, id2))
    return unmatched_idx


def main():
    print('Start generating positive dataset')
    dev_df = load_as_dataframe(os.path.join(g_load_path, 'dev_train_basic.csv'))
    coo_df = load_as_dataframe(os.path.join(g_load_path, 'cookie_all_basic.csv'))
    # uniq.pkl load
    uniq_ref = pickle.load(open(os.path.join(g_load_path, 'uniq.pkl')))
    uniq_ref_dev = padding(g_index_dummy.get('dev'), uniq_ref.get('dev'))
    uniq_ref_coo = padding(g_index_dummy.get('coo'), uniq_ref.get('coo'))
    # example generation, to get length
    dev_df_dummy_fill = add_dummies(dev_df.iloc[0],
                                    g_index_dummy.get('dev'),
                                    uniq_ref_dev)
    coo_df_dummy_fill = add_dummies(coo_df.iloc[0],
                                    g_index_dummy.get('coo'),
                                    uniq_ref_coo)
    if g_num_neg <= 0:
        g_num_neg = len(matched_idx)
    # start matching and unmatching
    matched_idx = get_match_idx(dev_df, coo_df)
    unmatched_idx = get_unmatch_idx(dev_df, coo_df, g_num_neg)
    # pos and neg data initialization
    pos_data = {'dev': sparse.csr_matrix((len(matched_idx), len(dev_df_dummy_fill))),
                'coo': sparse.csr_matrix((len(matched_idx), len(coo_df_dummy_fill)))}
    neg_data = {'dev': sparse.csr_matrix((g_num_neg, len(dev_df_dummy_fill))),
                'coo': sparse.csr_matrix((g_num_neg, len(coo_df_dummy_fill)))}
    # start filling pos/neg data
    for i, (x, y) in enumerate(matched_idx):
        pos_data['dev'][i] = add_dummies(dev_df.iloc[x])
        pos_data['coo'][i] = add_dummies(coo_df.iloc[x])
    print('positive data generating done.')
    for i, (x, y) in enumerate(unmatched_idx):
        neg_data['dev'][i] = add_dummies(dev_df.iloc[x])
        neg_data['coo'][i] = add_dummies(coo_df.iloc[x])
    print('negitive data generating done.')
    # saving
    # TODO remain a problem, I want to get a panacea here to avoid memory filling; that is, directly dumping csr_matrix to .csv file.



if __name__ == '__main__':
    main()
