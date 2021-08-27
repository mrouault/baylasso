import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter('ignore')
sns.set_style('darkgrid')

from Gibbs.BridgeNM import bridgenm
from Gibbs.BridgeTM import bridgetm
from Gibbs.essmcmc import essmcmc
from data.corr import gencorr

alpha = [k/200 for k in range(1, 199)]
alpha0 = np.array([1, 0.5, 0.2, 0.17, 0.001])
k0 = len(alpha0)
p = 15
n = 100

N = 1000
essnm = []
esstm = []
delta = np.array([0 if k%4 == 0 else np.random.randn() for k in range(p)])

for k in range(len(alpha)):
    X, Y = gencorr(n, p, alpha[k], k0, delta, alpha0)
    nm = bridgenm(X = X, Y = Y, alpha = 1.5).run(N)
    tm = bridgetm(X = X, Y = Y, alpha = 1.5).run(N)
    essnm.append(essmcmc(nm.beta[100:, ])[0])
    esstm.append(essmcmc(tm.beta[100:, ])[0])

fig, ax = plt.subplots()
ax.plot(alpha, esstm, color = "tab:blue", label = "Triangular mixture")
ax.plot(alpha, essnm, color = "tab:orange", label = "Normal mixture")
ax.set_xlabel("Effective Sample Size")
ax.set_ylabel("Correlation parameter")
fig.legend(loc = "upper right")
fig.show()
