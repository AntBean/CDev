#-*- coding: utf-8 -*-
# By Jake
"""
Model file.
"""

__all__ = ['Model']

import os
import sys
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
        # TODO
        params = ()
        for elem in self.sequential._list:
            params += (elem._coef,)

    def forward(self, input):
        """ forward through"""
        return self.sequential.forward(input)

    def fit(self, input, labels, args):
        """ fitting """
        # TODO args
        self.sequential.fit(input, label, args)

    def createLr(self):
        """create linear regression module"""
        # TODO

    def createSequential(self, config):
        self.sequential = Sequential()
        for elem_config in config:
            sequential.add(self.createModule(elem_config))
        return self.sequential

