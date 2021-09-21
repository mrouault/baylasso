import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from importance.importance_sampling import importance
from Gibbs.BridgeNM import bridgenm
from Gibbs.BridgeTM import bridgetm

import warnings
warnings.simplefilter('ignore')

sns.set_style('darkgrid')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)

N = 10000
alpha = 1.5
imp = importance(N = N, alpha = alpha, bridge = True).mv_gaussian(X, Y).weight()
nm = bridgenm(X = X, Y = Y, alpha = alpha).run(N)
tm = bridgetm(X = X, Y = Y, alpha = alpha).run(N)

fignm, axnm = plt.subplots()
sns.histplot(x = imp.beta[:, 0], weights = imp.w, stat = "density", kde = True,
        ax = axnm, label = "Importance sampling", color = "tab:blue", bins = 30)
sns.histplot( x= nm.beta[:, 0], stat = "density", kde = True,
        ax = axnm, label = "Normal mixture", color = "tab:orange")
figtm, axtm = plt.subplots()
sns.histplot(x = imp.beta[:, 0], weights = imp.w, stat = "density", kde = True,
        ax = axtm, label = "Importance sampling", color = "tab:blue", bins = 30)
sns.histplot( x = tm.beta[:, 0], stat = "density", kde = True,
        ax = axtm, label = "Triangular mixture", color = "tab:orange")
fignm.legend(loc = "right")
figtm.legend(loc = "right")

fignm.show()
figtm.show()
