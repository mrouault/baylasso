import numpy as np
from sklearn.preprocessing import StandardScaler


def gencorr(n, p, alpha, k, delta, alpha0) :

    alpha0 = alpha0/np.sum(alpha0**2)
    X = np.empty((n, p))
    V = np.random.randn(n, p)
    for i in range(k) :
        X[:, i] = V[:, i]
    for i in range(k, p):
        X[:, i] = np.sqrt(1-alpha**2)*V[:, i]
        for j in range(k) :
            X[:, i]+=alpha*alpha0[j]*X[:, j]
    eps = np.random.randn(n)
    Y = X.dot(delta)+eps
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = (Y-np.mean(Y))/np.std(Y)
    return X, Y
