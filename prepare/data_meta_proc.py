#-*- coding: utf-8 -*-
"""
Data preparation file.
By Stephen Lai

Datasets:

    cookie_all_basic.csv
    dev_test_basic.csv
    dev_train_basic.csv
    id_all_ip.csv
    id_all_property.csv
    ipagg_all.csv
    property_category.csv

This code does five things:
    1. read the files cookie_all_basic.csv and dev_train_basic.csv and remove all
        the entries that lack a user identifier (i.e. drawbridge_handle == -1).

    2. these two data set share some common feature name so I prefix them with
        'd_' for device features and 'c_' for cookie features.

    3. convert all the categoricals to dummies. And then inner join the two
        dataset together on the user id to get the positive dataset.
        Theoretically, it should generate all the possible positive data.

    4. randomly choose a mount of negative data point to compose the negative
        dataset

    5. output to csv. In the "../Data"

TODO or Problems:

    1.  [done, randomly choose amout of data] It may have infinite negative data,
        making the dataset hard to generate.
        A simple feasible way is to compare one by one until we get enough data.
        But will this influence the result of out experiments? (bias)

    2.  The data has quite a lot of missing value (-1), how to deal with them ?
"""


import numpy as np
import pandas as pd
import random as r
from sklearn.feature_extraction import DictVectorizer as DV


class Data(object):

    def __init__(self, path="../Data/"):
        if not path.endswith('/'):
            path += '/'
        self.data_path = path

    def string_2_int(self, df, attr):
        for attr_elem in attr:
            if attr_elem not in df.columns:
                continue
            df[attr_elem] = df[attr_elem].apply(lambda x: int(x[x.find('_')+1: ]))
        return df

    def categorical_2_dummy(self, df):
        """docstring for categorical_2_dummy"""
        df = df.applymap(str)
        ch_dict = df.T.to_dict().values() # This creates huge memory ~ 10Gb
        vec = DV(sparse=True) 
        ch_array = vec.fit_transform(ch_dict)
        '''
        ch_array = ch_array.astype('float16')
        # This step kills everything, only 9862926 of 57083328046 have values
        # Calling toarray(), it will have 2byte*57083328046=106Gb
        df_after = pd.DataFrame(ch_array, dtype='float16') 
        '''
        # One drawback of this, is SparseDataFrame doesn't support float32 or float16, which is a shame
        '''
        # TODO: Wondering the cheapest way to penetrate pandas.DataFrame; why don't we just wait for stackOverflow.
        # TODO: simple, csr_matrix -> sparseDataFrame, without releasing toarray(), is this the most space efficient way?
        # TODO issue on github
        df_after = pd.SparseDataFrame(index=df.index, columns=vec.get_featrue_names())
        for i in np.arange(ch_array.shape[0]):
            elem = pd.SparseSeries(ch_array[i].toarray().ravel())
            df_after.loc[[2]] = [elem]  # not implemendted error
        '''
        # New method
        df_after = pd.DataFrame(ch_array[:, 0].toarray().ravel()).to_sparse(0)
        for i in range(1, ch_array.shape[1]):
            df_after[i] = ch_array[:, i].toarray().ravel()
            if i % 1000 == 0:
                print('Finish: ' + str(i))
        return df_after

    def dev_data_processing(self):
        """
        docstring for dev_data_processing
        """
        device_df = pd.read_csv(self.data_path + "dev_train_basic.csv")
        device_df.columns = [u'd_drawbridge_handle',
                             u'device_id',
                             u'device_type',
                             u'device_os',
                             u'd_country',
                             u'd_anonymous_c0',
                             u'd_anonymous_c1',
                             u'd_anonymous_c2',
                             u'd_anonymous_5',
                             u'd_anonymous_6',
                             u'd_anonymous_7']
        device_df = device_df[device_df.d_drawbridge_handle != '-1']
        boolean_variables = ['d_anonymous_c0']
        categorical_variables = ['device_type',
                                 'device_os',
                                 'd_country',
                                 'd_anonymous_c1',
                                 'd_anonymous_c2']
        string_variables = [u'd_drawbridge_handle', u'device_id']
        # int_variables = ['d_anonymous_5', 'd_anonymous_6', 'd_anonymous_7']
        changed_var = boolean_variables + categorical_variables
        ch_df = device_df[changed_var]
        ch_df_after = self.categorical_2_dummy(ch_df)
        device_df = device_df.drop(changed_var, axis=1)
        device_df = self.string_2_int(device_df, string_variables)
        device_df = device_df.to_sparse(0)
        device_df = device_df.join(ch_df_after)
        return device_df

    def cookie_data_processing(self):
        """
        docstring for cookie_data_processing
        """
        cookie_df = pd.read_csv(self.data_path + "cookie_all_basic.csv")
        cookie_df.columns = [u'c_drawbridge_handle',
                             u'cookie_id',
                             u'computer_os_type',
                             u'computer_browser_version',
                             u'c_country',
                             u'c_anonymous_c0',
                             u'c_anonymous_c1',
                             u'c_anonymous_c2',
                             u'c_anonymous_5',
                             u'c_anonymous_6',
                             u'c_anonymous_7']
        cookie_df = cookie_df[cookie_df.c_drawbridge_handle != '-1']
        boolean_var = ['c_anonymous_c0']
        categorical_var = [u'computer_os_type',
                           u'computer_browser_version',
                           u'c_country',
                           u'c_anonymous_c1',
                           u'c_anonymous_c2']
        # int_var = [u'c_anonymous_5', u'c_anonymous_6', u'c_anonymous_7']
        string_variables = [u'c_drawbridge_handle', u'cookie_id']
        changed_var = boolean_var + categorical_var
        ch_df = cookie_df[changed_var]
        ch_df_after = self.categorical_2_dummy(ch_df)
        cookie_df = cookie_df.drop(changed_var, axis=1)
        cookie_df = self.string_2_int(cookie_df, string_variables)
        cookie_df = cookie_df.to_sparse(0)
        cookie_df = cookie_df.join(ch_df_after)
        return cookie_df

    def gen_pos_data(self, device_df, cookie_df):
        """docstring for gen_pos_data"""
        # TODO basically, this is not supported. pandas doesn't support SparseDataFrame, which is a shame!
        res_df = pd.merge(device_df, cookie_df, how='inner',
                          left_on='d_drawbridge_handle',
                          right_on='c_drawbridge_handle', sort=True)
        res_df['label'] = 1
        return res_df

    def gen_neg_data(self, device_df, cookie_df, num_of_records):
        """docstring for gen_neg_data"""
        FACTOR = 19

        device_length = len(device_df)
        device_index = [r.randint(1, FACTOR) for x in range(device_length)]
        device_df["merge_index"] = device_index

        cookie_length = len(cookie_df)
        cookie_index = [r.randint(1, FACTOR) for x in range(cookie_length)]
        cookie_df["merge_index"] = cookie_index

        res_df = pd.merge(device_df, cookie_df, how='inner',
                          on='merge_index', sort=True)

        res_df = res_df[res_df['d_drawbridge_handle'] != res_df['c_drawbridge_handle']]
        res_df = res_df.drop("merge_index", axis=1)
        res_length = min(num_of_records, len(res_df))
        res_df = res_df.iloc[:, :res_length]
        res_df["label"] = -1

        return res_df

# This new method uses a more intuitive way, just pick one by one, to genertate
# the negative data. No merge, no drop, so no memory error. But, it has another
# problem of having a few duplicated entries.
    def simple_neg_data_generate(self, device_df, cookie_df, number_of_records):
        """docstring for simple_neg_data_generate"""
        list_of_rows = []
        d_len = len(device_df)
        c_len = len(cookie_df)

        i=0
        while i<num_of_records:
            d_row = device_df.iloc[r.randint(0,d_len-1),:]
            c_row = cookie_df.iloc[r.randint(0,c_len-1),:]
            if (d_row.d_drawbridge_handle != c_row.c_drawbridge_handle) :
                new_row = pd.concat([d_row,c_row],axis=0)
                list_of_rows.append(new_row)
                i += 1

        res_df = pd.concat(list_of_rows, axis = 1).T
        res_df["label"] = -1
        return res_df

    def main(self):
        """docstring for main"""

        device_df = self.dev_data_processing()
        print('device done')
        cookie_df = self.cookie_data_processing()
        print('cookie done')

        print('Generating positive data.')
        pos_data = self.gen_pos_data(device_df, cookie_df)
        print('Saving positive data')
        pos_data.to_csv("../Data/pos_dev_cookie.csv")

        num_of_neg_data = 3000
        print('Generating negative data.')
        neg_data = self.simple_neg_data_generate(device_df, cookie_df, num_of_neg_data)
        print('Saving nagative data')
        neg_data.to_csv("../Data/neg_dev_cookie.csv")

if __name__ == '__main__':
    data = Data()
    data.main()
