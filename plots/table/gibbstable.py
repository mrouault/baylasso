import numpy as np
import scipy
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

from importance.importance_sampling import importance
from Gibbs.GibbsLasso import gibbslasso
from Gibbs.essmcmc import essmcmc
import data.data

dc = {"n" : {}, "p" : {}, "TimeIS" : {}, "ESSIS" : {}, "TimeGibbs" : {}, "Median(ESSGibbs)" : {}}
for dfname in data.data.dY.keys():
    X = data.data.dX[dfname]
    Y = data.data.dY[dfname]
    n, p = X.shape
    dc["p"][dfname] = p
    dc["n"][dfname] = n
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = (Y-np.mean(Y))/np.std(Y)
    imp = importance(N = 1000)
    timp0 = time.time()
    imp.mv_gaussian(X, Y).weight()
    timp = time.time()-timp0
    dc["ESSIS"][dfname] = imp.ess
    dc["TimeIS"][dfname] = timp
    gb = gibbslasso(X = X, Y = Y)
    cpug0 = time.time()
    try :
        gb.run(1000)
        cpug = time.time()-cpug0
        dc["TimeGibbs"][dfname] = cpug
        dc["Median(ESSGibbs)"][dfname] = np.median(essmcmc(gb.beta))
    except scipy.linalg.LinAlgError :
        dc["TimeGibbs"][dfname] = np.nan
        dc["Median(ESSGibbs)"][dfname] = np.nan

df = pd.DataFrame.from_dict(dc)
