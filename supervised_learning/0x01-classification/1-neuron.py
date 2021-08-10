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
