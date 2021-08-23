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

nvar = 1000
lamb = 1
imp = importance(N = 1000, lamb = lamb)
gibbs = gibbslasso(X = X, Y = Y, lamb = lamb)
impvar = np.empty(nvar)
gibbsvar = np.empty(nvar)

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
    impvar[i] = np.median([np.average(imp.beta[:, j], weights = imp.w) for j in range(p)])
    time0g = time.time()
    gibbs.run(ngibbs)
    time.time()-time0g
    gibbsvar[i] = np.median([np.mean(gibbs.beta[:, j]) for j in range(p)])

dat = pd.DataFrame({"method" : ['smc' if k < nvar else 'gibbs' for k in range(2*nvar)],
    "E[beta | Y]" : list(impvar)+list(gibbsvar)})
sns.boxplot(x = "method", y = "E[beta | Y]", data = da, showfliers = False)
