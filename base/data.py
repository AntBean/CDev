#-*- coding: utf-8 -*-
# By Jake


__all__ = ['DataSource']

import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import pickle
from functools import reduce


"""
 Some aux functions defined first
 - load_as_dataframe
 - add_dummies
 - padding
 - list operations ( add, subtract, diff_subtract)
 - prune
"""


def load_as_dataframe(fn, drop=[]):
    """load csv file into memory"""
    assert(isinstance(drop, list))
    df = pd.read_csv(fn)
    df = df.drop(df.columns[drop], axis=1)
    return df


def add_dummies(ds, idx, ref, label=False):
    """convert a data series to an array with dummy variables"""

    def to_dummies(string, ref):
        """find the position of string in ref, and derive dummies"""
        assert isinstance(ref, list)
        idx = ref.index(string)  # string must be in the ref
        arr = np.zeros(len(ref), dtype=np.uint16)
        arr[idx] = 1
        return arr

    import pdb; pdb.set_trace()
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


def list_add(a, b):
    """a+b"""
    res = map(lambda x: x[0]+x[1], zip(a,b))
    return list(res)


def list_sub(a, b):
    """a-b"""
    res = map(lambda x: x[0]-x[1], zip(a,b))
    return list(res)


def list_diff_sub(a, b):
    """diff, drop and subtract the offset"""
    aux = []
    for bb in b:
        elem = list( map(lambda x: x>bb, a) )
        aux.append(elem)
    aux = reduce(lambda x, y: list_add(x,y), aux)
    return list_sub(a, aux)


def prune(dc, drop):
    for d in drop:
        dc.pop(d)
    old_keys = list( dc.keys() )
    new_keys = list_diff_sub(old_keys, drop)
    dc = dict( zip(new_keys, dc.values()) )
    return dc


"""
 Main functionality
"""
class DataSource(object):
    """data source class, streaming data samples"""
    def __init__(self, index_dummy, index_drop, data_path):
        # args parsing
        self.index_dummy = index_dummy
        self.index_drop = index_drop
        self.data_path = data_path
        self.dev_train_data_path = os.path.join(data_path, 'dev_train_basic.csv')
        self.dev_test_data_path = os.path.join(data_path, 'dev_test_basic.csv')
        self.coo_data_path = os.path.join(data_path, 'cookie_all_basic.csv')
        # adjust index_drop
        self.index_drop['dev'] = list( map(lambda x: x-1, self.index_drop.get('dev')) )
        self.index_drop['coo'] = list( map(lambda x: x-1, self.index_drop.get('coo')) )
        # adjust index_dummy, and according to index_drop
        self.index_dummy['dev'] = list( map(lambda x: x-1, self.index_dummy.get('dev')) )
        self.index_dummy['dev'] = list( filter(lambda x: x not in self.index_drop.get('dev'),
                                               self.index_dummy.get('dev')) )
        self.index_dummy['dev'] = list_diff_sub(self.index_dummy.get('dev'), self.index_drop.get('dev'))
        self.index_dummy['coo'] = list( map(lambda x: x-1, self.index_dummy.get('coo')) )
        self.index_dummy['coo'] = list( filter(lambda x: x not in self.index_drop.get('coo'),
                                               self.index_dummy.get('coo')) )
        self.index_dummy['coo'] = list_diff_sub(self.index_dummy.get('coo'), self.index_drop.get('coo'))
        # load in data
        self.dev_df_train = load_as_dataframe(self.dev_train_data_path, self.index_drop.get('dev'))
        self.dev_df_test = load_as_dataframe(self.dev_test_data_path, self.index_drop.get('dev'))
        self.coo_df = load_as_dataframe(self.coo_data_path, self.index_drop.get('coo'))
        # get uniq.pkl
        uniq_ref = pickle.load(open(os.path.join(self.data_path, 'uniq.pkl'), 'rb'))
        # clear w.r.t index_drop
        self.uniq_ref_dev = prune(uniq_ref.get('dev'), self.index_drop.get('dev'))
        self.uniq_ref_coo = prune(uniq_ref.get('coo'), self.index_drop.get('coo'))
        # pad
        self.uniq_ref_dev = padding(len(self.dev_df_train.columns)-1, uniq_ref.get('dev'))
        self.uniq_ref_coo = padding(len(self.coo_df.columns)-1, uniq_ref.get('coo'))
        # get training indices
        data_idx = pickle.load(open(os.path.join(self.data_path, 'indices.pkl'), 'rb'))
        self.train_data_idx_pos = data_idx.get('pos')
        self.train_data_idx_neg = data_idx.get('neg')
        # get the length of dataset
        length_dev, length_coo = 0, 0
        while length_dev == 0:
            id = np.random.randint(len(self.dev_df_train['drawbridge_handle']))
            dev_df_dummy_fill = add_dummies(self.dev_df_train.iloc[id],
                                            self.index_dummy.get('dev'),
                                            self.uniq_ref_dev)
            length_dev = len(dev_df_dummy_fill) if dev_df_dummy_fill is not None else 0
        while length_coo == 0:
            id = np.random.randint(len(self.coo_df['drawbridge_handle']))
            coo_df_dummy_fill = add_dummies(self.coo_df.iloc[id],
                                            self.index_dummy.get('coo'),
                                            self.uniq_ref_coo)
            length_coo = len(coo_df_dummy_fill) if coo_df_dummy_fill is not None else 0
        self.length_dev = length_dev
        self.length_coo = length_coo

    def getNextBatch(self, batch_size):
        """for training"""
        # TODO is this gonna be faster?
        batch_dev_tmp = np.zeros((batch_size, self.length_dev))
        batch_coo_tmp = np.zeros((batch_size, self.length_coo))
        # start filling the batch
        i = 0
        while i < batch_size-2:
            # One positive
            batch_dev_tmp_elem, batch_coo_tmp_elem = None, None
            while batch_dev_tmp_elem is not None or batch_coo_tmp_elem is not None:
                # get id
                pos_id = self.train_data_idx_pos[np.random.randint(len(self.train_data_idx_pos))]
                batch_dev_tmp_elem = add_dummies(self.dev_df_train.iloc[pos_id[0]],
                                                 self.index_dummy.get('dev'),
                                                 self.uniq_ref_dev)
                batch_coo_tmp_elem = add_dummies(self.coo_df.iloc[pos_id[1]],
                                                 self.index_dummy.get('coo'),
                                                 self.uniq_ref_coo)
            batch_dev_tmp[i,:], batch_coo_tmp[i,:] = batch_dev_tmp_elem, batch_coo_tmp_elem
            # One negative
            batch_dev_tmp_elem, batch_coo_tmp_elem = None, None
            while batch_dev_tmp_elem == None or batch_coo_tmp_elem == None:
                # get id
                neg_id = self.train_data_idx_neg[np.random.randint(len(self.train_data_idx_neg))]
                batch_dev_tmp_elem = add_dummies(self.dev_df_train.iloc[neg_id[0]],
                                                 self.index_dummy.get('dev'),
                                                 self.uniq_ref_dev)
                batch_coo_tmp_elem = add_dummies(self.coo_df.iloc[neg_id[1]],
                                                 self.index_dummy.get('coo'),
                                                 self.uniq_ref_coo)
            batch_dev_tmp[i+1,:], batch_coo_tmp[i+1,:] = batch_dev_tmp_elem, batch_coo_tmp_elem
            i += 2
        batch = {}
        batch['dev'] = sparse.csr_matrix(batch_dev_tmp)
        batch['coo'] = sparse.csr_matrix(batch_coo_tmp)
        return batch
