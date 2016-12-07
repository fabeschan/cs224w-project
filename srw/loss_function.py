import numpy as np
import transition_matrix

def squared_hinge_loss(x):
    return x ** 2 if x > 0.0 else 0.0

def squared_hinge_loss_derivative(x):
    return 2 * x if x > 0.0 else 0.0

def total_loss(p, removed, ROOT):
    s = 0.0
    # for (i,j)=(l,d), l not in removed, where d in removed
    for i in set(range(len(p))) - set(removed[ROOT]):
        for j in set(removed[ROOT]):
            s += squared_hinge_loss(p[i] - p[j])
    return s

