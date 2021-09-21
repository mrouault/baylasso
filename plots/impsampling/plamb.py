import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from importance.tempered_smc import TemperedImportance

sns.set_style('darkgrid')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape

N = 10000
temp = TemperedImportance(X = X, Y = Y, plot = True)
temp.run(N)

