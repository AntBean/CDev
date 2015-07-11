#-*- coding: utf-8 -*-
# By Jake

import os
import sys
import numpy as np
import linecache

# Some global variables
g_index_dummy = {'dev': [2, 3, 4, 5, 7, 8],
                 'coo': [2, 3, 4, 5, 7, 8]}
g_load_path = '../../Data'
g_save_path = '../../Data/2015.07.04.primitive'
g_num_neg = 0


"""
 Some aux functions defined first
 - add_dummies
 - padding
"""

def add_dummies(ds, idx, ref, label=False):
    """convert a data series to an array with dummy variables"""

    def to_dummies(string, ref):
        """find the position of string in ref, and derive dummies"""
        assert isinstance(ref, list)
        idx = ref.index(string)  # string must be in the ref
        arr = np.zeros(len(ref), dtype=np.uint16)
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

"""
 Main functionality
"""
class DataSource(object):
    """data source class, streaming data samples"""
    def __init__(self, index_dummy, data_path):
        self.index_dummy = index_dummy
        self.data_path = data_path
        self.dev_train_data_path = os.path.join(data_path, 'dev_train_basic.csv')
        self.dev_test_data_path = os.path.join(data_path, 'dev_test_basic.csv')
        self.coo_data_path = os.path.join(data_path, 'cookie_all_basic.csv')

    def getSingleSample(self, index, item='dev', set='train'):
        if item == 'dev':
            if set == 'train':
                this_path = self.dev_train_data_path
            elif set == 'test':
                this_path = self.dev_test_data_path
            else:
                print('set must be [train | test]', file=sys.stderr)
                sys.exit()
        elif item == 'coo':
            this_path = self.coo_data_path
        else:
            print('item must be [dev | coo]', file=sys.stderr)
            sys.exit()

        
        



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
    # save both matched and unmatched indices
    indices = {'pos': matched_idx, 'neg': unmatched_idx}
    os.system('mkdir -p ' + g_save_path)
    with open(os.path.join(g_save_path, 'indices.pkl'), 'wb') as idf:
        pickle.dump(indices, idf)
        idf.close()
    print('matched and unmatched indices generated and saved to ' + os.path.join(g_save_path, 'indices.pkl'))
    print('saving done')
