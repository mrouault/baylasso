import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

from importance.importance_sampling import importance
from Gibbs.GibbsLasso import gibbslasso
from data.corr import gencorr

alpha = 0.7
p = 15
n = 200
alpha0 = np.array([1])
k = len(alpha0)
delta = np.array([0 if c%4 == 0 else np.random.randn() for c in range(p)])
X, Y = gencorr(n, p, alpha, k, delta, alpha0)

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

