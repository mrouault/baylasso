import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns

from data.corr import gencorr

sns.set_style('darkgrid')

lamb = 1

alpha = 0
p = 10
n = [100, 1000, 10000, 100000]
alpha0 = np.array([1])
k = len(alpha0)
delta = np.array([0 if c%4 == 0 else np.random.randn() for c in range(p)])
a = np.arange(-1, 1, 1e-4).size
beta = np.empty((a, p))
for j in range(p) :
    beta[:, j] = np.arange(-1, 1, 1e-4)

fig, ax = plt.subplots()
figl, axl = plt.subplots()

for ni in n :
    X, Y = gencorr(ni, p, alpha, k, delta, alpha0)
    C = np.linalg.cholesky(X.T.dot(X))
    Cmean = scipy.linalg.solve_triangular(C, X.T.dot(Y), lower = True)
    mean = scipy.linalg.solve_triangular(C.T, Cmean, lower = False)
    sig2 = 1/ni * np.sum((Y-X.dot(mean))**2)

    loglik = np.empty(a)
    logprior = np.empty(a)
    for i in range(a) :
        root = C.T.dot(beta[i, :] - mean)
        loglik[i] = (-0.5/sig2 * root.T.dot(root))
        logprior[i] = -lamb/np.sqrt(sig2) * np.sum(np.abs(beta[i, :]))

    lik = np.exp(loglik - np.max(loglik))
    lik = lik/sum(lik)
    prior = np.exp(logprior - np.max(logprior))
    prior = prior/sum(prior)

    ax.plot(beta[:, 0], lik, label = "n = "+str(ni))
    axl.plot(beta[:, 0], prior, label = "n = "+str(ni))

fig.legend(loc = "upper right")
figl.legend(loc = "upper right")
ax.set_xlim((-0.1, 0.1))
axl.set_xlim((-0.1, 0.1))

fig.show()
figl.show()

