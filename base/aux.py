#-*- coding: utf-8 -*-
# By Jake
'''
aux file, defining Module and Sequential
'''

__all__ = ['Module', 'Sequential']

import sys
from sklearn import linear_model
from copy import copy

class Module(object):
    """Module class"""
    def __init__(self, module):
        self._module = module
        self.output = None
        self.input = None
        # Get a function handle for fitting
        #if isinstance(self._module, CountVectorizer):
        #    self.fit = self._module.fit_transform
        #    self.forward = self._module.transform
        if isinstance(self._module, linear_model.logistic.LogisticRegression):
            # TODO  
            # too ugly can be better though
            self.fit = self._module.fit
            self.forward = self._module.predict
        else:
            print('The module added is not supported.')
            sys.exit(-1)

    def getModule(self):
        return self._module


class Sequential(object):
    """Primitive class Sequential"""
    def __init__(self):
        self._list = list()

    def __len__(self):
        """get length"""
        return len(self._list)

    def __getitem__(self, index):
        """get value"""
        return self._list[index]

    def __setitem__(self, index, module_elem):
        """set module_elem"""
        self._list[index] = module_elem

    def __str__(self):
        """Overwrite __str__ for print"""
        return ("{}\n"*len(self._list)).format(*self._list)

    def __contrains__(self, elem):
        return elem in self._list
    
    def add(self, module_elem):
        """add a new module"""
        self._list.append(Module(module_elem))

    def fit(self, input, labels):
        # TODO this is just too verbose and ugly
        """fitting the sequential"""

    def forward(self, input, normalization=True):
        # TODO this is just too verbose and ugly
        """forwarding"""
        return self._list[-1].output
