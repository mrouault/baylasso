import numpy as np
import scipy
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

from importance.importance_sampling import importance
from Gibbs.BridgeNM import bridgenm
from Gibbs.BridgeTM import bridgetm
from Gibbs.essmcmc import essmcmc
import data.data

alpha = 1.5
dc = {"n" : {}, "p" : {}, "TimeIS" : {}, "ESSIS" : {}, "TimeNM" : {}, 
        "Median(ESSNM)" : {}, "TimeTM" : {}, "Median(ESSTM)" : {}}

for dfname in data.data.dY.keys():
    X = data.data.dX[dfname]
    Y = data.data.dY[dfname]
    n, p = X.shape
    dc["p"][dfname] = p
    dc["n"][dfname] = n
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = (Y-np.mean(Y))/np.std(Y)
    imp = importance(N = 1000, alpha = alpha, bridge = True)
    timp0 = time.time()
    imp.mv_gaussian(X, Y).weight()
    timp = time.time()-timp0
    dc["ESSIS"][dfname] = imp.ess
    dc["TimeIS"][dfname] = timp
    nm = bridgenm(X = X, Y = Y, alpha = alpha)
    cpunm0 = time.time()
    nm.run(1000)
    cpunm = time.time()-cpunm0
    dc["TimeNM"][dfname] = cpunm
    dc["Median(ESSNM)"][dfname] = np.median(essmcmc(nm.beta))
    tm = bridgetm(X = X, Y = Y, alpha = alpha)
    cputm0 = time.time()
    try :
        tm.run(1000)
        cputm = time.time()-cputm0
        dc["TimeTM"][dfname] = cputm
        dc["Median(ESSTM)"][dfname] = np.median(essmcmc(tm.beta))
    except scipy.linalg.LinAlgError :
        dc["TimeTM"][dfname] = np.nan
        dc["Median(ESSTM)"][dfname] = np.nan

df = pd.DataFrame.from_dict(dc)
