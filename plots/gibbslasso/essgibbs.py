import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from Gibbs.GibbsLasso import gibbslasso
from Gibbs.essmcmc import essmcmc

sns.set_style('darkgrid')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape

C = np.linalg.cholesky(X.T.dot(X))
Cmean = scipy.linalg.solve_triangular(C, X.T.dot(Y), lower = True)
mean = scipy.linalg.solve_triangular(C.T, Cmean, lower = False)
sig2 = 1/n * np.sum((Y-X.dot(mean))**2)

N = 1000+100
lamb = [k/10 for k in range(1, 200)]
ess = []
esssig2 = []
beta = gibbslasso(X = X, Y = Y, lamb = 1, sig2 = sig2)
betasig2 = gibbslasso(X = X, Y = Y, lamb = 1)
for k in range(len(lamb)):
    beta.lamb = lamb[k]
    betasig2.lamb = lamb[k]
    beta.run(N)
    betasig2.run(N)
    ess.append(essmcmc(beta.beta[100:, ]))
    esssig2.append(essmcmc(betasig2.beta[100:, ]))

fig, ax = plt.subplots()
figsig2, axsig2 = plt.subplots()

ax.plot(lamb, ess)
ax.set_xlabel("lambda")
ax.set_ylabel("ESS")

axsig2.plot(lamb, esssig2)
axsig2.set_xlabel("lambda")
axsig2.set_ylabel("ESS")

fig.show()
figsig2.show()
