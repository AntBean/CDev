# -*- coding: utf-8 -*-
# By Jake
"""
Main file for the whole pipeline
"""

import os
import _pickle
import numpy as np
from data import *
from model import *
from config import config


class Main(object):
    """The Main class"""
    def __init__(self, config):
        self.config = config

    def new(self):
        # Loading data
        # TODO
        '''
        # Await Mr. Stephen
        print("==> Loading data")
        data = Data(self.config.get('data'))
        print("==> Shuffling data")
        data.shuf()
        self.train_data = {'data': data.getTrainData(), 'title': data.getTrainTitle(), 'label': data.getTrainLabel()}
        self.test_data = {'data': data.getTestData(), 'title': data.getTestTitle(), 'label': data.getTestLabel()}
        '''
        self.record = {'train_error': 0 ,'test_error': 0}
        # Loading Model
        print("==> Loading model")
        self.model = Model(config.get('model'))

    def train(self, config):
        # Training
        print("==> Start training")
        # TODO
        '''
        # Await Mr. Stephen
        data_ensemble = [self.train_data.get('data'), self.train_data.get('title')]
        self.model.fit(data_ensemble, self.train_data.get('label'))
        '''
        print("==> Training ends")

    def test(self):
        # Testing
        print("==> Start testing")
        # TODO
        '''
        # Await Mr. Stephen
        data_ensemble = [self.test_data.get('data'), self.test_data.get('title')]
        test_pred = self.model.forward(data_ensemble)
        test_pred_acc = np.where(self.test_data.get('label') == test_pred)[0].shape[0] / len(self.test_data.get('label'))
        self.record['test_error'] = 1 - test_pred_acc
        # Safe way
        data_ensemble = [self.train_data.get('data'), self.train_data.get('title')]
        train_pred = self.model.forward(data_ensemble)
        train_pred_acc = np.where(self.train_data.get('label') == train_pred)[0].shape[0] / len(self.train_data.get('label'))
        self.record['train_error'] = 1 - train_pred_acc
        self.record['Num_ngrams'] = len(self.model.getVocabulary())
        print(self.record)
        '''

    def save(self, config):
        # Saving
        print("==> Saving models")
        os.system('mkdir -p ' + config.get('save'))
        filename = os.path.join(config.get('save'), config.get('name'))
        _pickle.dump({'config': self.config, 'record': self.record},
                      open(filename+'_main', 'wb'))
        _pickle.dump(self.model.getSequense(),
                     open(filename+'_sequence', 'wb'),
                     protocol=4)
        print("==> Saving done.")

    def main(self):
        main.new()
        main.train(self.config.get('train'))
        main.test(self.config.get('test'))
        main.save(self.config.get('main'))

if __name__ == '__main__':
    main = Main(config)
    main.main() 
