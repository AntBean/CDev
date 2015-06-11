# -*- coding: utf-8 -*-
# By Jake
'''
aux file, defining Module and Sequential
'''

__all__ = ['Module', 'Sequential']


class Module(object):
    """Module class"""
    def __init__(self, module):
        self._module = module
        self.output = None
        self.input = None
        # TODO exceptions are await.
        self.fit = self._module.fit
        self.forward = self._module.predict

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

    def fit(self, input, labels, fit_flag):
        """fitting the sequential"""
        assert len(fit_flag) == len(self._list)
        for flag, elem in zip(fit_flag, self._list):
            if not flag:
                # Not fitting the model, but forwarding
                input = elem.forward(input)
            else:
                # Fitting the module
                elem.fit(input, labels)
                input = elem.forward(input)

    def forward(self, input, normalization=True):
        """forwarding"""
        for elem in self._list:
            elem.output = elem.forward(input)
        return self._list[-1].output

    def getParameters(self):
        """get the parameters of modules of self._list
           concat them in a tuple structure"""
        params = ()
        for elem in self._list:
            if hasattr(elem, '_coef'):
                params += (elem._coef,)
            else:
                params += (None,)
        return params

    def getSequense(self):
        """get the whole list of models"""
        return self._list
