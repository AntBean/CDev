#-*- coding: utf-8 -*-
"""
Data file.
By Stephen Lai

Datasets:

    cookie_all_basic.csv
    dev_test_basic.csv
    dev_train_basic.csv
    id_all_ip.csv
    id_all_property.csv
    ipagg_all.csv
    property_category.csv

"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV

class Data(object):

    def __init__(self, path = "../../KuggleData/"):
        self.data_path = path

    def categorical_2_dummy(df):
        """docstring for categorical_2_dummy"""
        df = df.applymap(str)
        ch_dict = df.T.to_dict().values()
        vec = DV(sparse = False)
        ch_array = vec.fit_transform(ch_dict)
        df_after = pd.DataFrame(ch_array)
        dummy_columns = vec.get_feature_names()
        df_after.columns = dummy_columns
        df_after.index = device_df.index
        return df_after

    def dev_data_processing(self):
        """docstring for dev_data_procssing
            * categorical to dummy

        """

        device_df = pd.read_csv(self.data_path + "dev_train_basic.csv")

        boolean_variables = ['anonymous_c0']
        categorical_variables = ['device_type', 'device_os','country',
                                    'anonymous_c1', 'anonymous_c2']
        int_variables = ['anonymous_5', 'anonymous_6', 'anonymous_7']
        changed_var = boolean_variables + categorical_variables
        ch_df = device_df[changed_var]

        ch_df_after = categorical_2_dummy(ch_df)

        device_df = device_df.drop(changed_var,axis=1)
        device_df = device_df.join(ch_df_after)
        return device_df

    def cookie_data_processing(self):
        """docstring for cookie_data_processing"""
        cookie_df = pd.read_csv(self.data_path + "cookie_all_basic.csv")

        # TODO : finish the rest of this
        pass

    def gen_pos_data(self):
        """docstring for gen_pos_data"""
        pass

    def gen_neg_data(self):
        """docstring for gen_neg_data"""
        pass

    def main(self):
        """docstring for main"""



