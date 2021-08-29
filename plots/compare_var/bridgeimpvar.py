import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from importance.importance_sampling import importance
from Gibbs.BridgeNM import bridgenm
from Gibbs.BridgeTM import bridgetm

sns.set_style('darkgrid')

import warnings
warnings.simplefilter('ignore')

alpha = 1.5
X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape

nvar = 100
imp = importance(N = 1000, alpha = alpha, bridge = True)
nm = bridgenm(X = X, Y = Y, alpha = alpha)
tm = bridgetm(X = X, Y = Y, alpha = alpha)

impestnm = np.empty((nvar, p))
impesttm = np.empty((nvar, p))
nmest = np.empty((nvar, p))
tmest = np.empty((nvar, p))

ntm = 1000
nnm = 1000
cpuimp0 = time.time()
imp.mv_gaussian(X, Y).weight()
cpuimp = time.time()-cpuimp0
cpunm0 = time.time()
nm.run(nnm)
cpunm = time.time()-cpunm0
cputm0 = time.time()
tm.run(ntm)
cputm = time.time()-cputm0

imp.N = int(nnm * cpunm/cpuimp)

for i in range(nvar):
    imp.mv_gaussian(X, Y).weight()
    for j in range(p):
        impestnm[i, j] = np.average(imp.beta[:, j], weights = imp.w)
    nm.run(nnm)
    for j in range(p):
        nmest[i, j] = np.mean(nm.beta[:, j])

imp.N = int(ntm * cputm/cpuimp)

for i in range(nvar):
    imp.mv_gaussian(X, Y).weight()
    for j in range(p):
        impesttm[i, j] = np.average(imp.beta[:, j], weights = imp.w)
    tm.run(ntm)
    for j in range(p):
        tmest[i, j] = np.mean(tm.beta[:, j])

tempnvar = np.array([np.var(impestnm[:, j]) for j in range(p)])
nmvar = np.array([np.var(nmest[:, j]) for j in range(p)])

temptvar = np.array([np.var(impesttm[:, j]) for j in range(p)])
tmvar = np.array([np.var(tmest[:, j]) for j in range(p)])

fig, ax = plt.subplots()
dat = pd.DataFrame({"mixture" : ['Normal' if k < p else 'Triangular' for k in range(2*p)],
    "log10(VarBridge/VarIS)" : list(np.log10(nmvar/tempnvar))+list(np.log10(tmvar/temptvar))})
sns.boxplot(x = "mixture", y = "log10(VarBridge/VarIS)", data = dat, ax = ax)
fig.show()
