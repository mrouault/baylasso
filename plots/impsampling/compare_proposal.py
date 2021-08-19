import numpy as np
import scipy
from scipy.stats import laplace
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns

from importance.importance_sampling import importance

sns.set_style("darkgrid")


def laplace_proposal(m, p, lamb, sig2) :
    beta = np.empty(shape = (m, p))
    for j in range(p) :
        beta[:, j] = laplace.rvs(scale = np.sqrt(sig2)/lamb, size = m)
    return beta

def laplace_reweight(m, beta, X, Y, sig2) :
    lw = np.empty(m)
    for i in range(m):
        zi = Y-X.dot(beta[i, :])
        lw[i] = -0.5/sig2 *  zi.T.dot(zi)
    w = np.exp(lw-np.max(lw))
    w/= sum(w)
    ess = sum(w)**2/sum(w**2)
    return w, ess

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
n, p = X.shape
Y = (Y-np.mean(Y))/np.std(Y)
C = np.linalg.cholesky(X.T.dot(X))
Cmean = scipy.linalg.solve_triangular(C, X.T.dot(Y), lower = True)
mean = scipy.linalg.solve_triangular(C.T, Cmean, lower = False)
sig2 = 1/n * np.sum((Y-X.dot(mean))**2)

betanorm = importance(N = 10000, lamb = 1, sig2 = sig2).mv_gaussian(X, Y).weight()

betalap = laplace_proposal(10000, p, 1, sig2)
wlap, esslap = laplace_reweight(10000, betalap, X, Y, sig2)

fig, ax = plt.subplots()
sns.histplot(x = betanorm.beta[:, 0], weights = betanorm.w, stat = "density", 
        bins = 30, ax = ax, color = "tab:blue")
figl, axl = plt.subplots()
sns.histplot(x = betalap[:, 0], weights = wlap, stat = "density",
        bins = 100, ax = axl, color = "tab:orange")

fig.show()
figl.show()

