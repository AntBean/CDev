#-*- coding: utf-8 -*-
# By Jake

import os
import pickle
import subprocess

# global variables
g_index_dummy = {'dev': [2, 3, 4, 5, 7, 8],
                 'coo': [2, 3, 4, 5, 7]}
g_load_path = '../../DataSample'
g_save_path = '../../DataSample'


def get_uniq_column(fn, idx):
    '''opens a sub-thread to get uniq strings'''
    p = subprocess.Popen(['./get_uniq.sh', str(idx), fn],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.split('\n')
    out.pop()  # last one is usually empty string
    return out


def main():
    uniq_list = {'dev': {}, 'coo': {}}
    for idx in g_index_dummy.get('dev'):
        uniq_list['dev'][idx] = get_uniq_column(os.path.join(g_load_path,
                                                'dev_train_basic.csv'), idx)
    for idx in g_index_dummy.get('coo'):
        uniq_list['coo'][idx] = get_uniq_column(os.path.join(g_load_path,
                                                'cookie_all_basic.csv'), idx)
    # saving
    save = open(os.path.join(g_save_path, 'uniq.pkl'), 'w')
    pickle.dump(uniq_list, save)
    save.close()


if __name__ == '__main__':
    main()
