import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from importance.importance_sampling import importance
from Gibbs.GibbsLasso import gibbslasso

sns.set_style('darkgrid')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape

nvar = 100
lamb = 1
imp = importance(N = 1000, lamb = lamb)
gibbs = gibbslasso(X = X, Y = Y, lamb = lamb)
impest = np.empty((nvar, p))
gibbsest = np.empty((nvar, p))

ngibbs = 1000
cpuimp0 = time.time()
imp.mv_gaussian(X, Y).weight()
cpuimp = time.time()-cpuimp0
cpug0 = time.time()
gibbs.run(ngibbs)
cpug = time.time()-cpug0

imp.N = int(ngibbs * cpug/cpuimp)

for i in range(nvar):
    time0imp = time.time()
    imp.mv_gaussian(X, Y).weight()
    time.time()-time0imp
    for j in range(p):
        impest[i, j] = np.average(imp.beta[:, j], weights = imp.w)
    time0g = time.time()
    gibbs.run(ngibbs)
    time.time()-time0g
    for j in range(p):
        gibbsest[i, j]  = np.mean(gibbs.beta[:, j])


tempvar = np.array([np.var(impest[:, j]) for j in range(p)])
gibbsvar = np.array([np.var(gibbsest[:, j]) for j in range(p)])

fig, ax = plt.subplots()
sns.boxplot(x = np.log10(gibbsvar/tempvar),ax = ax)
fig.show()

for j in range(p):
    fig, ax = plt.subplots()
    dat = pd.DataFrame({"method" : ['IS' if k < nvar else 'Gibbs' for k in range(2*nvar)],
        "E[beta | Y]" : list(impest[:, j])+list(gibbsest[:, j])})
    sns.boxplot(x = "method", y = "E[beta | Y]", data = dat, showfliers = False, ax = ax)
    fig.show()
