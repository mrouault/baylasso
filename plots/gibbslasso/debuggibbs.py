import numpy as np
from scipy.stats import laplace
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from Gibbs.GibbsLasso import gibbslasso

sns.set_style('darkgrid')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape

N = 10000

beta = gibbslasso(X = X, Y = Y, debug = True, klambda = 100, thetalambda = 10)
beta.run(N)

lap = np.empty(N)
for i in range(N):
    lap[i] = laplace.rvs(size = 1, scale = np.sqrt(beta.sig2[i])/beta.lamb[i])

fig, ax = plt.subplots()
sns.histplot(x = beta.beta[1000:, 0], stat = "density", kde = True,
        label = "Gibbs sampling (lasso)", color = "tab:blue", ax = ax, bins = 50)
sns.histplot(x = lap, stat = "density", kde = True,
        color = "tab:red", ax = ax, label = "Laplace", bins = 50)
fig.legend(loc = "upper right")
fig.show()

figtrace, axtrace = plt.subplots()
axtrace.plot(beta.beta[1000:, 0])
figtrace.show()

