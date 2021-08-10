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
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
