import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from importance.importance_sampling import importance

sns.set_style('darkgrid')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape

N = 10000
lamb = [k/10 for k in range(1, 200)]
ess1 = []
ess2 = []
ess3 = []
ess4 = []
beta = importance(N = 10000, lamb = 1, bridge = True).mv_gaussian(X, Y)
for k in range(len(lamb)):
    beta.lamb = lamb[k]
    beta.alpha = 0.5
    beta.weight()
    ess1.append(beta.ess)
    beta.alpha = 1
    beta.weight()
    ess2.append(beta.ess)
    beta.alpha = 1.5
    beta.weight()
    ess3.append(beta.ess)
    beta.alpha = 2
    beta.weight()
    ess4.append(beta.ess)

fig, ax = plt.subplots()

ax.plot(lamb, ess1, label = r"$\alpha = 0.5$")
ax.plot(lamb, ess2, label = r"$\alpha = 1$")
ax.plot(lamb, ess3, label = r"$\alpha = 1.5$")
ax.plot(lamb, ess4, label = r"$\alpha = 2$")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("ESS")

fig.legend(loc = "upper right")
fig.show()
