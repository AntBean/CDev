#-*- coding: utf-8 -*-
# By Jake
"""
Model file.
"""

__all__ = ['Model']

import numpy as np
from sklearn import linear_model
from aux import *


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
        params = ()
        for elem in self.sequential._list:
            if hasttr(elem, '_coef'):
                params += (elem._coef,)
            else:
                params += (None,)

    def forward(self, input):
        """ forward through"""
        return self.sequential.forward(input)

    def fit(self, input, labels, args):
        """ fitting """
        # TODO args
        self.sequential.fit(input, label, args)

    def createLR(self, args):
        """create linear regression module"""
        return linear_model.LinearRegression(args)

    def createLogR(self, args):
        """create logistic regression module"""
        return linear_model.LogisticRegression(args)

    def createSequential(self, config):
        self.sequential = Sequential()
        for elem_config in config:
            sequential.add(self.createModule(elem_config))
        return self.sequential
