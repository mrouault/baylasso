import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler


import data.data
from importance.tempered_smc import TemperedImportance
from Gibbs.GibbsLasso import gibbslasso

sns.set_style('darkgrid')

nvar = 100
for dfname in ["turbine2014", "temperature", "carbon", "wave_adelaide"]:
    X = data.data.dX[dfname]
    Y = data.data.dY[dfname]
    scaler  = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = (Y-np.mean(Y))/np.std(Y)
    n, p = X.shape
    print(dfname)
    temp = TemperedImportance(X = X, Y = Y, verbose = False, plot = True)
    gb = gibbslasso(X = X, Y = Y)

    impest = np.empty((nvar, p))
    gibbsest = np.empty((nvar, p))

    ngibbs = 1000
    cpuimp0 = time.time()
    temp.run(ngibbs)
    cpuimp = time.time()-cpuimp0
    cpug0 = time.time()
    gb.run(ngibbs)
    cpug = time.time()-cpug0
    Nimp = int(ngibbs * cpug/cpuimp)

    for i in range(nvar):
        temp.run(Nimp)
        for j in range(p):
            impest[i, j] = np.average(temp.beta[:, j], weights = temp.W)
        gb.run(ngibbs)
        for j in range(p):
            gibbsest[i, j] = np.mean(gb.beta[:, j])

    tempvar = np.array([np.var(impest[:, j]) for j in range(p)])
    gibbsvar = np.array([np.var(gibbsest[:, j]) for j in range(p)])

    fig, ax = plt.subplots()
    sns.boxplot(x = np.log10(gibbsvar/tempvar), ax = ax)
    fig.show()


