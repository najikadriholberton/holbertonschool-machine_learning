#!/usr/bin/env python3
"""Neuron Module"""
import numpy as np


class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """Init the neuron"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """forward propagation"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Logistic Regression Cost Function"""
        m = Y.shape[1]
        Z = (Y*np.log(A)+(1-Y)*np.log(1.0000001-A))
        return -np.sum(Z) / m

    @property
    def W(self):
        """Get Weight"""
        return self.__W

    @property
    def b(self):
        """Get Bias"""
        return self.__b

    @property
    def A(self):
        """Get Activation"""
        return self.__A
