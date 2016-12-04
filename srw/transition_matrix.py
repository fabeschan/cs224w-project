import numpy as np
import loss_function

def f_ext(u, v, kb):
    x = [1, 0, 0.5]
    return np.array(x)

def activation_function(x_uv, w):
    return np.exp(np.dot(w, x_uv))

def feature_matrix(g, f_ext)
    x = np.zeros((g.GetNodes(), g.GetNodes()))

    for u_NI in g.Nodes():
        u = u_NI.GetId()
        for v_NI in g.Nodes():
            v = v_NI.GetId()
            x_uv = f_ext.calculate(u, v)
            x[u, v] = x_uv
    return x

def edge_weights(w, fm, activation_function):
    ''' return matrix of edge weights no normalization '''
    # fm: the calculated feature_matrix
    v = np.vectorize(activation_function, excluded=['w'])
    A = v(fm, w=w)
    return A

def normalize(A):
    n = np.sum(A)
    return A / n

def transition_matrix(w, fm):
    return normalize(edge_weights(w, fm, activation_function))

def pagerank_one_iter(p, w, fm, alpha=0.85):
    tm = transition_matrix(w, fm)
    return (1 - alpha) * np.matul(p, tm) + alpha * p

def p_grad_w(p, w, fm, delta=10**-12):
    fp = pagerank_one_iter
    grad = zeros((len(p), len(w))
    for pi in range(len(p)):
        for wi in range(len(wi)):
            ww = np.copy(w)
            ww[wi] += delta
            grad[pi][wi] = (fp(p, ww, fm) - fp(p, w, fm)) / delta
   return grad

def j_grad_w(p, w, fm):
    pgw = p_grad_w(p, w, fm)
    grad = zeros([len(w)])
    for wi in range(len(w)):
        s = 0
        for i in range(len(p)):
            for j in range(len(p)):
                s += loss_function.squared_hinge_loss_derivative(p[i] - p[j]) * (pgw[i][wi] - pgw[j][wi])
        grad[wi] = s
    return grad

def gradient_descent_step(p, w, fm, r=0.01):
    new_w = w - r * j_grad_w(p, w, fm)
    return new_w


