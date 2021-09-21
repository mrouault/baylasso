import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from importance.tempered_smc import TemperedImportance
import data.data

sns.set_style('darkgrid')

names = ["blog", "aquatic_toxicity"]

N = 10000
for dfname in names :
    X = data.data.dX[dfname]
    Y = data.data.dY[dfname]
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = (Y-np.mean(Y))/np.std(Y)

    temp = TemperedImportance(X = X, Y = Y, plot = True, verbose = False, step = 1e-3)
    temp.run(N)
