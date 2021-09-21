import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import data.data
from importance.tempered_smc import TemperedImportance
plot = {"x" : [], "y" : [], "label" : []}

sns.set_style('darkgrid')
for dfname in data.data.dY.keys() :
    X = data.data.dX[dfname]
    Y = data.data.dY[dfname]
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = (Y-np.mean(Y))/np.std(Y)
    n, p = X.shape
    temp = TemperedImportance(X = X, Y = Y, verbose = False)
    temp.run(10000, len_chain = 100)

    if not dfname in ["blog", "indoor"] :
        plot["x"].append(p)
        plot["label"].append(dfname)
        plot["y"].append(temp.temp_alg.t)
    else :
        print(dfname, "Niter = ", temp.temp_alg.t)

plt.figure(figsize = (15, 10))
plt.xlabel('Feature dimension', fontsize = 15)
plt.ylabel('Number of Tempering iterations', fontsize = 15)
plt.scatter(plot["x"], plot["y"], marker = 'o')
for labels, x, y in zip(plot["label"], plot["x"], plot["y"]):
    plt.annotate(labels, xy = (x, y))
plt.show()
