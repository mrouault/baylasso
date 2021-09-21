import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler

from importance.tempered_smc import TemperedImportance
from Gibbs.BridgeNM import bridgenm
from Gibbs.BridgeTM import bridgetm
import data.data

sns.set_style('darkgrid')
import warnings
warnings.simplefilter('ignore')

alpha = 1.5
nvar = 100
for dfname in ["student-mat", "superconduct"] :
    X = data.data.dX[dfname]
    Y = data.data.dY[dfname]
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = (Y-np.mean(Y))/np.std(Y)
    n, p = X.shape
    imp = TemperedImportance(X = X, Y = Y, step = 1e-3, verbose = False, alpha = alpha, bridge = True)
    imp.run(1000, len_chain = 100)
    timeimp = imp.temp_alg.cpu_time
    lamb = imp.lamb
    nm = bridgenm(X = X, Y = Y, alpha = alpha, nu = lamb)
    cpunm0 = time.time()
    nm.run(1000)
    cpunm = time.time()-cpunm0

    Nm = int(1000*cpunm/timeimp)
    
    tm = bridgetm(X = X, Y = Y, alpha = alpha, nu = lamb)
    cputm0 = time.time()
    tm.run(1000)
    cputm = time.time()-cputm0

    Tm = int(1000*cputm/timeimp)

    impestm = np.empty((nvar, p))
    impesnm = np.empty((nvar, p))
    nmest = np.empty((nvar, p))
    tmest = np.empty((nvar, p))
    print(Nm)
    print(Tm)

    for i in range(nvar):
        imp.run(Nm, len_chain = 100)
        for j in range(p):
            impesnm[i, j] = np.average(imp.beta[:, j], weights = imp.W)
        nm.run(1000)
        for j in range(p):
            nmest[i, j] = np.mean(nm.beta[:, j])

        imp.run(Tm, len_chain = 100)
        for j in range(p):
            impestm[i, j] = np.average(imp.beta[:, j], weights = imp.W)
        tm.run(1000)
        for j in range(p):
            tmest[i, j] = np.mean(tm.beta[:, j])

    tempnvar = np.array([np.var(impesnm[:, j]) for j in range(p)])
    nmvar = np.array([np.var(nmest[:, j]) for j in range(p)])

    temptvar = np.array([np.var(impestm[:, j]) for j in range(p)])
    tmvar = np.array([np.var(tmest[:, j]) for j in range(p)])

    fig, ax = plt.subplots()
    dat = pd.DataFrame({"mixture" : ['Normal' if k <p else 'Triangular' for k in range(2*p)],
        "log10(VarBridge/VarSMC)" : list(np.log10(nmvar/tempnvar))+list(np.log10(tmvar/temptvar))})
    sns.boxplot(x = "mixture", y = "log10(VarBridge/VarSMC)", data = dat, ax = ax)
    fig.show()
    
