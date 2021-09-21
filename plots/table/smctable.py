import numpy as np
import scipy
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

from importance.tempered_smc import TemperedImportance
from Gibbs.GibbsLasso import gibbslasso
import data.data

nvar = 100
dc = {"n" : {}, "p" : {}, "SMCSize" : {}, "NIter" : {}, "lambda" : {}, "Median(log10(VarGibbs/VarSMC))" : {}}
for dfname in data.data.dY.keys():    
    X = data.data.dX[dfname]
    Y = data.data.dY[dfname]
    n, p = X.shape
    dc["p"][dfname] = p
    dc["n"][dfname] = n
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = (Y-np.mean(Y))/np.std(Y)
    temp = TemperedImportance(X = X, Y = Y, verbose = False)
    temp.run(1000, len_chain = 100)
    timetemp = temp.temp_alg.cpu_time
    dc["lambda"][dfname] = temp.lamb
    gb = gibbslasso(X = X, Y = Y, lamb = temp.lamb)
    cpug0 = time.time()
    try : 
        gb.run(1000)
        cpug = time.time() - cpug0
        Ntemp = int(1000*cpug/timetemp)
        dc["SMCSize"][dfname] = Ntemp
        impest = np.empty((nvar, p))
        gibbsest = np.empty((nvar, p))
        for i in range(nvar):
            temp.run(Ntemp, len_chain = 100)
            if i ==0 :
                dc["NIter"][dfname] = temp.temp_alg.t
            for j in range(p):
                impest[i, j] = np.average(temp.beta[:, j], weights = temp.W)
            gb.run(1000)
            for j in range(p):
                gibbsest[i, j] = np.nanmean(gb.beta[:, j])
        tempvar = np.array([np.var(impest[:, j]) for j in range(p)])
        gibbsvar = np.array([np.var(gibbsest[:, j]) for j in range(p)])
        dc["Median(log10(VarGibbs/VarSMC))"][dfname] = np.median(np.log10(gibbsvar/tempvar))
    except scipy.linalg.LinAlgError :
        dc["SMCSize"][dfname] = 1000
        dc["NIter"][dfname] = temp.temp_alg.t
        dc["Median(log10(VarGibbs/VarSMC))"][dfname] = np.nan

df = pd.DataFrame.from_dict(dc)
