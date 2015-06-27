#-*- coding:utf-8 -*-
# By Jake

'''
This file provides some aux functions for pandas.SparseDataFrame
'''

__all__ = ['sparse_merge']


import numpy as np
import pandas as pd


def sparse_merge(dfl, dfr, left_on, right_on, how='inner', in_place=True):
    """merge function for two pd.SparseDataFrames
    # Stupid, brute-force implementation of the merge
    # Space effiecient, and in place

    input: dfl, dfr, left_on, right_on, how, in_place
    return: df, merged dataframe
    """

    # Assertions
    assert isinstance(dfl, pd.SparseDataFrame)
    assert isinstance(dfr, pd.SparseDataFrame)
    assert left_on in dfl.columns
    assert right_on in dfr.columns

    # left dafaframe is also the shorter one
    if len(dfl) > len(dfr):
        temp = dfl
        dfl = dfr
        dfr = temp

    # Generate a merge list, and ready to merge
    elem_chunk_list = []
    for idxl, eleml in enumerate(dfl[left_on]):
        # left dataframe is shorter
        for idxr, elemr in enumerate(dfr[right_on]):
            if eleml == elemr:
                # get one more pair
                elem_chunk_list.append((idxl, idxr))
                # TODO bijective?
                break
            else:
                continue

    # Merge, based on two dataframes are all numeric
    dfl_idx_elem = list(map(lambda x: x[0], elem_chunk_list))
    dfr_idx_elem = list(map(lambda x: x[1], elem_chunk_list))
    # take out two matrices
    mat_l = dfl.iloc[dfl_idx_elem].values
    mat_r = dfr.drop(right_on, axis=1).iloc[dfr_idx_elem].values
    # Concat them
    mat_result = np.hstack((mat_l, mat_r))
    df_result = pd.DataFrame(mat_result).to_sparse()
    return df_result
