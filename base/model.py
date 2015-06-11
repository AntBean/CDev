#-*- coding: utf-8 -*-
# By Jake
"""
Model file.
"""

__all__ = ['Model']

from sklearn import linear_model
from aux import *

name_handle = {'LR': linear_model.LinearRegression,
              'LogR': linear_model.LogisticRegression}


class Model(object):
    """Model class"""
    def __init__(self, config):
        self.config = config
        self.sequential = self.createSequential(self.config)

    def __str__(self):
        """print function, overwrite __str__ function"""
        return self.sequential.__str__

    def getModel(self):
        """get sequential"""
        return self.sequential

    def get(self, idx):
        """torch style, take out component"""
        return self.sequential[idx]

    def getParameters(self):
        """get the parameters of sequential"""
        return self.sequential.getParameters()

    def forward(self, input):
        """ forward through"""
        return self.sequential.forward(input)

    def fit(self, input, labels, fit_flag=None):
        """ fitting """
        if fit_flag is None:
            fit_flag = [False]*len(self.sequential)
            fit_flag[-1] = True
        self.sequential.fit(input, label, fit_flag)

    def createModule(self, config):
        """create module"""
        if config.has_key('args') and config.get('args') != None:
            return name_handle.get(config.get('name'))(config.get('args'))
        else:
            return name_handle.get(config.get('name'))()

    def createSequential(self, config):
        self.sequential = Sequential()
        for elem_config in config:
            sequential.add(self.createModule(elem_config))
        return self.sequential
