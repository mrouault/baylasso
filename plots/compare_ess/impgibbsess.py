import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

from importance.importance_sampling import importance
from Gibbs.GibbsLasso import gibbslasso
from data.corr import gencorr
from Gibbs.essmcmc import essmcmc

alpha = [k/200 for k in range(1, 199)]
alpha0 = np.array([1, 0.5, 0.2, 0.17, 0.001])
k0 = len(alpha0)
p = 15
n = 100
lamb = 1
N = 1000

ess = []
essgibbs = []
delta = np.array([0 if k%4 == 0 else np.random.randn() for k in range(p)])

for k in range(len(alpha)):
    X, Y = gencorr(n, p, alpha[k], k0, delta, alpha0)
    imp = importance(N = N, lamb = lamb).mv_gaussian(X, Y).weight()
    ess.append(imp.ess)

    gibbs = gibbslasso(X = X, Y = Y, lamb = lamb).run(N + 100)
    essgibbs.append(essmcmc(gibbs.beta[100:, ])[0])

fig, ax = plt.subplots()
ax.plot(alpha, ess, color = "tab:blue", label = "Importance sampling")
ax.plot(alpha, essgibbs, color = "tab:orange", label = "Gibbs sampling")
ax.set_xlabel("Correlation parameter")
ax.set_ylabel("Effective Sample Size")
fig.legend(loc = "upper right")
fig.show()
