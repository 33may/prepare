import numpy as np


class SGD:
    def update(self, w, d_w, learning_rate):
        return w - d_w * learning_rate


class MomentumSGD:
    '''
    Implements Momentum SGD update
    '''
    def __init__(self, momentum=0.9):
        self.momentum = 0.9
        self.velocities = {}
    
    def update(self, w, d_w, learning_rate, param_name):
        '''
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        '''

        if param_name not in self.velocities:
            self.velocities[param_name] = np.zeros_like(w)

        self.velocities[param_name] = self.momentum * self.velocities[param_name] - learning_rate * d_w

        return w + self.velocities[param_name]
