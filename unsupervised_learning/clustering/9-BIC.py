'''
Write a function def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False): that finds the best number of clusters for a GMM using the Bayesian Information Criterion:
'''

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    if kmax is None:
        kmax = len(X)
    if kmin < 1 or kmax < kmin or not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None, None, None
    l = []
    b = []
    for k in range(kmin, kmax + 1):
        pi, m, S = expectation_maximization(X, k, iterations, tol, verbose)
        p = k * d * (d + 1) / 2 + k - 1
        n = len(X)
        l.append(log_likelihood(X, pi, m, S))
        b.append(p * np.log(n) - 2 * l[-1])
    best_k = np.argmin(b) + kmin
    best_result = (pi, m, S)
    return best_k, best_result, np.array(l), np.array(b)
