import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from importance.importance_sampling import importance
from Gibbs.GibbsLasso import gibbslasso

sns.set_style('darkgrid')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)

N = 10000
lamb = 1
imp = importance(N = 10000, lamb = lamb).mv_gaussian(X, Y).weight()
gibbs = gibbslasso(X = X, Y = Y, lamb = lamb).run(N+100)

fig, ax = plt.subplots()
sns.histplot(x = imp.beta[:, 0], weights = imp.w, stat = "density", kde = True,
        ax = ax, label = "Importance sampling", color = "tab:blue", bins = 30)
sns.histplot(x = gibbs.beta[100:, 0], stat = "density", kde = True, ax = ax, 
        label = "Gibbs sampling", color = "tab:orange", bins = 30)
ax.set_title("Histograms")
fig.legend(loc = "right")

figcdf, axcdf = plt.subplots()
sns.ecdfplot(x = imp.beta[:, 0], weights = imp.w, label = "Importance sampling",
        ax = axcdf, color = "tab:blue")
sns.ecdfplot(x = gibbs.beta[100:, 0], label = "Gibbs sampling", color = "tab:orange", 
        ax = axcdf)
axcdf.set_title("Empirical CDF")
figcdf.legend(loc = "right")

fig.show()
figcdf.show()
