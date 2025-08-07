
"""
Gaussian Process Regression
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap



def make_data():
    model = lambda x: x * np.sin(x)
    x = np.array([1, 3, 5, 6, 8])
    x = np.random.random(np.random.randint(3, 9)) * 10
    return x[:, np.newaxis], model(x)



class Kernel(ABC):
    def __init__(self, **kwargs):
        self.hyperparameters = kwargs
    
    @abstractmethod
    def __call__(self, x1, x2):
        "compute the kernel similarity between the two inputs"
        


class RBF(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, x1, x2):
        assert all(hasattr(x, '__len__') for x in (x1,x2)) and len(x1)==len(x2)
        v = self.hyperparameters.get('signal_variance')
        l = self.hyperparameters['length']
        gamma = 1 / (2 * l ** 2)
        return v * np.exp(-gamma * sum((x1-x2)**2 for x1,x2 in zip(x1, x2)))



class GaussianProcessRegression:
    def __init__(self, kernel, noise=0.0):
        self.kernel = kernel
        self.noise = noise
        
    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y)
        self.K = self._compute_K(X, X)
        I = self.noise * np.eye(self.K.shape[0])  # add noise to diagonal
        self.K_inverse = np.linalg.inv(self.K + I) 
        return self

    def optimize(self, **grid):
        X, y = self.X, self.y.reshape(-1, 1)
        d = OrderedDict(sorted(grid.items()))
        mx = -np.inf
        
        for e in product(*d.values()):
            self.kernel.hyperparameters = {k:v for k,v in zip(d.keys(), e)}
            K = self._compute_K(X, X)
            n = K.shape[0]
            loglikelihood = -(1/2) * y.T @ np.linalg.inv(K) @ y \
                            -(1/2) * np.log(np.linalg.det(K))\
                            -(n/2) * np.log(2*np.pi)
            loglikelihood = loglikelihood[0][0]
            
            if loglikelihood > mx:
                mx = loglikelihood
                params = self.kernel.hyperparameters
        # Set the best params
        self.kernel.hyperparameters = params
        self.fit(X, y)
        return self
    
    def predict(self, X):
        X = np.array(X)

        K1 = self._compute_K(self.X, X)  #K*
        K2 = self._compute_K(X, X)  #K**
        
        y_mu = K1.T @ self.K_inverse @ self.y.reshape(-1, 1)
        variance = K2 - K1.T @ self.K_inverse @ K1
        return y_mu.flatten(), np.diagonal(variance)
        
    def _compute_K(self, X1, X2):
        return np.array([[self.kernel(x1, x2) for x2 in X2] for x1 in X1])



##### DEMO ####

# Make toy data
X, y = make_data()

# Define a kernel and model
kernel = RBF(signal_variance=2.0, length=1.0)
model = GaussianProcessRegression(kernel=kernel, noise=0)

# Fit
model.fit(X, y)

# Optimize the hypers
grid = {'signal_variance': np.logspace(-2, 2, 5),
        'length': np.logspace(-2, 1, 4),}
model.optimize(**grid)

# Predict
x_new = np.linspace(start=0, stop=10, num=100)
X_new = x_new.reshape(-1, 1)

y_pred, var_pred = model.predict(X_new)
std_pred = var_pred ** (1/2)

# Vizualize
plt.plot(X.flatten(), y, 'ok')
plt.plot(x_new, y_pred, 'b-')
plt.fill_between(x_new, y_pred - std_pred, y_pred + std_pred, alpha=0.5)
plt.title(f"optimized hyperparameters:\n{model.kernel.hyperparameters}")
