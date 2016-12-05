import snap
import numpy as np

def adjacency_matrix(g):
    A = np.zeros([g.GetNodes(), g.GetNodes()])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = 1.0 if g.IsEdge(i, j) else 0
    return A
