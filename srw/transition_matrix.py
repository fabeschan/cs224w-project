import numpy as np
import loss_function
import features

def activation_function(x_uv, w):
    return np.exp(-np.dot(x_uv, w))

def feature_matrix(g, fx):
    x = np.zeros((g.GetNodes(), g.GetNodes(), fx.NUM_FEATURES))

    for u_NI in g.Nodes():
        u = u_NI.GetId()
        for v_NI in g.Nodes():
            v = v_NI.GetId()
            x_uv = fx.feature_vector(u, v)
            x[u][v] = x_uv
    return x

def edge_weights(w, fm, activation_function):
    ''' return matrix of edge weights no normalization '''
    # fm: the calculated feature_matrix
    A = np.zeros([fm.shape[0], fm.shape[1]])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = activation_function(fm[i][j], w)
    return A

def normalize(A):
    n = np.sum(A)
    return A / n

def normalize_rows(A):
    m = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        m[i,:] = normalize(A[i,:])
    return m

def transition_matrix(w, fm, am):
    '''
    w: weights
    fm: feature matrix
    am: adjacency matrix
    '''
    return normalize(edge_weights(w, fm, activation_function) * am)

def pagerank_one_iter(p, w, fm, am, alpha=0.15):
    tm = transition_matrix(w, fm, am)
    #print 'tm:', tm
    p_new = (1 - alpha) * np.matmul(p, tm.T) + alpha * p
    #print 'matmul:', np.matmul(p, tm.T)
    return p_new

def p_grad_w(p, w, fm, am, delta=10**-12):
    grad = np.zeros((len(p), len(w)))
    opr = pagerank_one_iter(p, w, fm, am)
    for wi in range(len(w)):
        ww = np.copy(w)
        ww[wi] += delta
        grad[:,wi] = (pagerank_one_iter(p, ww, fm, am) - opr) / delta
    return grad

def j_grad_w(p, w, fm, am, removed, ROOT):
    pgw = p_grad_w(p, w, fm, am)
    grad = np.zeros([len(w)])
    for wi in range(len(w)):
        s = 0

        # for (i,j)=(l,d), where d in removed, l not in removed
        for i in set(removed[ROOT]):
            for j in set(range(len(p))) - set(removed[ROOT]):
                s += loss_function.squared_hinge_loss_derivative(p[i] - p[j]) * (pgw[i][wi] - pgw[j][wi])
        grad[wi] = s
    return grad

def gradient_descent_step(p, w, fm, am, removed, ROOT, r=0.1):
    new_w = w - r * j_grad_w(p, w, fm, am, removed, ROOT)
    return new_w


