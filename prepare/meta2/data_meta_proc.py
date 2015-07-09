# -*- coding:utf-8-*-
# By Jake

import os
import numpy as np
from scipy import sparse
import pandas as pd
import pickle
import gc


# Some global variables
g_index_dummy = {'dev': [2, 3, 4, 5, 7, 8],
                 'coo': [2, 3, 4, 5, 7, 8]}
g_load_path = '../../Data'
g_save_path = '../../Data/2015.07.04.primitive'
g_num_neg = 0


def load_as_dataframe(fn):
    """load csv file into memory"""
    df = pd.read_csv(fn)
    return df


def add_dummies(ds, idx, ref, label=False):
    """convert a data series to an array with dummy variables"""

    def to_dummies(string, ref):
        """find the position of string in ref, and derive dummies"""
        assert isinstance(ref, list)
        idx = ref.index(string)  # string must be in the ref
        arr = np.zeros(len(ref), )
        arr[idx] = 1
        return arr

    assert len(ref) == len(ds)-1  # references have to be padded, and drawbridge_handle is the only diff
    assert isinstance(ds, pd.core.series.Series)
    assert isinstance(idx, list)
    arr = ds.values.copy()
    # completeness examining
    incomplete = False
    for elem in arr:
        incomplete = incomplete or elem=='-1'
    if incomplete:
        return
    offset_dummies = 0
    # converting
    for idx_elem in idx:
        idx_elem_offset = idx_elem + offset_dummies
        raw = arr[idx_elem_offset]
        arr = np.delete(arr, idx_elem_offset)
        dummies = to_dummies(raw, ref[idx_elem])
        arr = np.insert(arr, idx_elem_offset, dummies)
        offset_dummies += len(dummies) - 1
    if not label:
        arr = arr[1:]
    return arr


def padding(num, ref, pad=[]):
    """padding the references"""
    assert isinstance(ref, dict)
    assert isinstance(num, int)
    diff = list( set(range(num)) - set(ref.keys()) ) 
    for i in diff:
        ref[i] = pad 
    return ref


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


def prune_lil_matrix(mat, idx):
    """prune a lil matrix one row, i.e. No idx-th row"""
    assert isinstance(mat, sparse.lil_matrix)
    mat.rows = np.delete(mat.rows, idx)
    mat.data = np.delete(mat.data, idx)
    mat._shape = (mat._shape[0]-1, mat._shape[1])
    return mat


def prune_allzero_row_lil_matrix(mat):
    """find and prune all zero rows in lil_matrix"""
    rowidx, _ = mat.nonzero()
    rowidx = list( set(range(mat.shape[0])) - set(rowidx) )
    offset = 0
    for relem in rowidx:
        mat = prune_lil_matrix(mat, relem-offset)
        offset += 1
    return mat


def main():
    dev_df = load_as_dataframe(os.path.join(g_load_path, 'dev_train_basic.csv'))
    coo_df = load_as_dataframe(os.path.join(g_load_path, 'cookie_all_basic.csv'))
    # uniq.pkl load
    uniq_ref = pickle.load(open(os.path.join(g_load_path, 'uniq.pkl'), 'rb'))
    uniq_ref_dev = padding(len(dev_df.columns)-1, uniq_ref.get('dev'))
    uniq_ref_coo = padding(len(coo_df.columns)-1, uniq_ref.get('coo'))
    # adjust g_index_dummy
    g_index_dummy['dev'] = list( map(lambda x: x-1, g_index_dummy.get('dev')) )
    g_index_dummy['coo'] = list( map(lambda x: x-1, g_index_dummy.get('coo')) )
    # example generation, to get length
    length_dev, length_coo = 0, 0
    while length_dev == 0:
        id = np.random.randint(len(dev_df['drawbridge_handle']))
        dev_df_dummy_fill = add_dummies(dev_df.iloc[id],
                                        g_index_dummy.get('dev'),
                                        uniq_ref_dev)
        length_dev = len(dev_df_dummy_fill) if dev_df_dummy_fill is not None else 0
    while length_coo == 0:
        id = np.random.randint(len(coo_df['drawbridge_handle']))
        coo_df_dummy_fill = add_dummies(coo_df.iloc[id],
                                        g_index_dummy.get('coo'),
                                        uniq_ref_coo)
        length_coo = len(coo_df_dummy_fill) if coo_df_dummy_fill is not None else 0
    # start matching and unmatching
    matched_idx = get_match_idx(dev_df, coo_df)
    global g_num_neg
    if g_num_neg <= 0:
        g_num_neg = len(matched_idx)
    unmatched_idx = get_unmatch_idx(dev_df, coo_df, g_num_neg)
    print('matched and unmatched indices generated.')
    # pos and neg data initialization
    pos_data = {'dev': sparse.lil_matrix((len(matched_idx), length_dev)),
                'coo': sparse.lil_matrix((len(matched_idx), length_coo))}
    neg_data = {'dev': sparse.lil_matrix((g_num_neg, length_dev)),
                'coo': sparse.lil_matrix((g_num_neg, length_coo))}
    # start filling pos/neg data
    for i, (x, y) in enumerate(matched_idx):
        pos_dev_elem = add_dummies(dev_df.iloc[x], g_index_dummy.get('dev'), uniq_ref_dev)
        pos_coo_elem = add_dummies(coo_df.iloc[y], g_index_dummy.get('coo'), uniq_ref_coo)
        if pos_dev_elem is not None and pos_coo_elem is not None:
            continue
        pos_data['dev'][i,:] = pos_dev_elem   # Feeling suspicious about this line; might not be space efficient. Learn Sparse matrix construction!!
        pos_data['coo'][i,:] = pos_coo_elem
        del pos_dev_elem
        del pos_coo_elem
        gc.collect()
    print('positive data generating done.')
    for i, (x, y) in enumerate(unmatched_idx):
        neg_dev_elem = add_dummies(dev_df.iloc[x], g_index_dummy.get('dev'), uniq_ref_dev)
        neg_coo_elem = add_dummies(coo_df.iloc[y], g_index_dummy.get('coo'), uniq_ref_coo)
        if neg_dev_elem is not None and neg_coo_elem is not None:
            continue
        neg_data['dev'][i,:] = neg_dev_elem
        neg_data['coo'][i,:] = neg_coo_elem
        del neg_dev_elem
        del neg_coo_elem
        gc.collect()
    print('negitive data generating done.')
    # pruning the all-zero line
    prune_allzero_row_lil_matrix(pos_data['dev'])
    prune_allzero_row_lil_matrix(pos_data['coo'])
    prune_allzero_row_lil_matrix(neg_data['dev'])
    prune_allzero_row_lil_matrix(neg_data['coo'])
    # unit-test
    assert(pos_data['dev'].shape[0] == pos_data['coo'].shape[0])
    assert(neg_data['dev'].shape[0] == neg_data['coo'].shape[0])
    print('No assertion is raised.')
    # saving, by pickle
    os.system('mkdir -p ' + g_save_path)
    with open(os.path.join(g_save_path, 'pos_data.pkl'), 'wb') as psf:
        pickle.dump(pos_data, psf)
        psf.close()
    with open(os.path.join(g_save_path, 'neg_data.pkl'), 'wb') as nsf:
        pickle.dump(neg_data, nsf)
        nsf.close()
    print('saving done')
    

if __name__ == '__main__':
    main()
