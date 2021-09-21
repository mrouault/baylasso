import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from importance.tempered_smc import TemperedImportance
from Gibbs.GibbsLasso import gibbslasso


sns.set_style('darkgrid')
N = 10000
X, Y = load_boston(return_X_y = True)    
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape
temp = TemperedImportance(X = X, Y = Y, step = 1e-2, plot = True)
temp.run(N, len_chain = 100)
gb = gibbslasso(X = X, Y = Y, lamb = temp.temp_alg.fk.lamb).run(N)
fig, ax = plt.subplots()
sns.histplot(x = temp.beta[:, 0], weights = temp.W, stat = "density", kde = True,
            color = "tab:blue", bins = 30, ax = ax, label = "SMC")
sns.histplot(x = gb.beta[:, 0], stat = "density", kde = True, color = "tab:orange", 
            ax = ax, bins = 30, label = "Gibbs sampler")
fig.legend(loc = "upper right")
fig.show()

