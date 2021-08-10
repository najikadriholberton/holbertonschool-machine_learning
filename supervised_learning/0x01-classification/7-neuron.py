#!/usr/bin/env python3
"""Neuron Module"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaulate the Neuron"""
        A = self.forward_prop(X)
        c = self.cost(Y, A)
        p = np.where(A > 0.5, 1, 0)
        return p, c

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate the gradient descent for the neuron"""
        Z = self.cost(Y, A)
        dz = A - Y
        dw = (1/len(Y[0])) * np.matmul(dz, X.T)
        db = (1/len(Y[0])) * np.sum(dz)
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """train the model and return the evaluation"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        iters = []
        for i in range(iterations + 1):
            A, cost = self.evaluate(X, Y)
            if i != iterations:
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A, alpha)
            if verbose:
                if i % step == 0 or i == 0 or i == iterations:
                    print("Cost after {} iterations: {}".format(i, cost))
                    costs.append(cost)
                    iters.append(i)
        if graph:
            plt.plot(iters, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.xlim(0, iterations)
            plt.show()
        return A, cost

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
