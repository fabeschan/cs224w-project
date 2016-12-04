import numpy as np
import transition_matrix

def squared_hinge_loss(x):
    return x ** 2 if x > 0 else 0

def squared_hinge_loss_derivative(x):
    return 2 * x if x > 0 else 0

'''
def evaluate(self, p, w):
    # evaluate the total loss where p is the pagerank vector and w the weight vector
    shl = np.vectorize(self.squared_hinge_loss)
    t = shl(np.dot(p.T, w))
    return np.sum(t)
'''
