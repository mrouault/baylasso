import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler

from importance.tempered_smc import TemperedImportance
from Gibbs.GibbsLasso import gibbslasso
import data.data

sns.set_style('darkgrid')

l = ['student-mat', 'superconduct']
for dfname in l :
    X = data.data.dX[dfname]
    Y = data.data.dY[dfname]
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y  = (Y-np.mean(Y))/np.std(Y)
    n, p = X.shape

    nvar = 100
    temp = TemperedImportance(X = X, Y = Y, step = 1e-3, verbose = False)
    temp.run(1000, len_chain = 100)
    timetemp = temp.temp_alg.cpu_time
    lamb = temp.lamb
    gibbs = gibbslasso(X = X, Y = Y, lamb = lamb)
    cpug0 = time.time()
    gibbs.run(1000)
    cpug = time.time()-cpug0

    Ntemp = int(1000*cpug/timetemp)
    impest = np.empty((nvar, p))
    gibbsest = np.empty((nvar, p))

    for i in range(nvar):
        temp.run(Ntemp, len_chain = 100)
        for j in range(p):
            impest[i, j] = np.average(temp.beta[:, j], weights = temp.W)
        gibbs.run(1000)
        for j in range(p):
            gibbsest[i, j] = np.mean(gibbs.beta[:, j])

    tempvar = np.array([np.var(impest[:, j]) for j in range(p)])
    gibbsvar = np.array([np.var(gibbsest[:, j]) for j in range(p)])

    fig, ax = plt.subplots()
    sns.boxplot(x = np.log10(gibbsvar/tempvar), ax = ax)
    fig.show()
