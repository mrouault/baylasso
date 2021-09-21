import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time
import warnings
from sklearn.preprocessing import StandardScaler

from Gibbs.BridgeNM import bridgenm
from Gibbs.BridgeTM import bridgetm
from importance.tempered_smc import TemperedImportance
import data.data
from Gibbs.essmcmc import essmcmc


sns.set_style('darkgrid')
warnings.simplefilter('ignore')
alpha = 1.5
nvar = 100

for dfname in ["turbine2014", "temperature", "carbon", 
        "wave_adelaide"]:
        X = data.data.dX[dfname]
        Y = data.data.dY[dfname]
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        Y = (Y-np.mean(Y))/np.std(Y)
        n, p = X.shape
        print(dfname)
        temp = TemperedImportance(X = X, Y = Y, verbose = False, plot = True, bridge = True, alpha = alpha)
        nm = bridgenm(X = X, Y = Y, alpha = alpha)
        tm = bridgetm(X = X, Y = Y, alpha = alpha)

        impestnm = np.empty((nvar, p))
        impesttm = np.empty((nvar, p))
        nmest = np.empty((nvar, p))
        tmest = np.empty((nvar, p))
        ntm = 1000
        nnm = 1000
        temp.run(1000)
        cpuimp = temp.temp_alg.cpu_time
        cpunm0 = time.time()
        nm.run(nnm)
        cpunm = time.time()-cpunm0
        cputm0 = time.time()
        tm.run(ntm)
        cputm = time.time()-cputm0

        Nimp = int(nnm * cpunm/cpuimp)

        for i in range(nvar):
            temp.run(Nimp)
            for j in range(p):
                impestnm[i, j] = np.average(temp.beta[:, j], weights = temp.W)
            nm.run(nnm)
            for j in range(p):
                nmest[i, j] = np.mean(nm.beta[:, j])

        Nimp = int(ntm * cputm/ cpuimp)

        for i in range(nvar):
            temp.run(Nimp)
            for j in range(p):
                impesttm[i, j] = np.average(temp.beta[:, j], weights = temp.W)
            tm.run(ntm)
            for j in range(p):
                tmest[i, j] = np.mean(tm.beta[:, j])
        tempnvar = np.array([np.var(impestnm[:, j]) for j in range(p)])
        nmvar = np.array([np.var(nmest[:, j]) for j in range(p)])

        temptvar = np.array([np.var(impesttm[:, j]) for j in range(p)])
        tmvar = np.array([np.var(tmest[:, j]) for j in range(p)])

        fig, ax = plt.subplots()
        dat = pd.DataFrame({"mixture" : ['Normal' if k< p else 'Triangular' for k in range(2*p)],
            "log10(VarBridge/VarIS)" : list(np.log10(nmvar/tempnvar))+list(np.log10(tmvar/temptvar))})
        sns.boxplot(x = "mixture", y = "log10(VarBridge/VarIS)", data = dat, ax = ax)
        fig.show()
        
