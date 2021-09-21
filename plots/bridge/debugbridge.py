import numpy as np
from scipy.stats import gennorm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import warnings

from Gibbs.BridgeNM import bridgenm
from Gibbs.BridgeTM import bridgetm

sns.set_style('darkgrid')
warnings.simplefilter('ignore')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape

N = 10000+100
alpha = 1.5

nm = bridgenm(X = X, Y = Y, debug = True, klambda = 20, thetalambda = 1, ksig2 = 10, thetasig2 = 10, alpha = alpha)
tm = bridgetm(X = X, Y = Y, debug = True, klambda = 20, thetalambda = 1, ksig2 = 10, thetasig2 = 10, alpha = alpha)
nm.run(N)
tm.run(N)

lapgennm = np.empty(N)
lapgentm = np.empty(N)
for i in range(N):
    lapgennm[i] = gennorm.rvs(alpha, size = 1, scale = nm.nu[i]**(-1/alpha))
    lapgentm[i] = gennorm.rvs(alpha, size = 1, scale = tm.nu[i]**(-1/alpha))
fignm, axnm = plt.subplots()
figtm, axtm = plt.subplots()

sns.histplot(x = nm.beta[100:, 0], stat = "density", kde = True,
        label = "Normal mixture", color = "tab:blue", ax =axnm)
sns.histplot(x = lapgennm[100:], stat = "density", kde = True, 
        label = "Generalized Laplace prior", color = "tab:red", ax = axnm)

sns.histplot(x = tm.beta[100:, 0], stat = "density", kde = True,
        label = "Triangular mixture", color = "tab:blue", ax = axtm)
sns.histplot(x = lapgentm[100:], stat = "density", kde = True,
        label = "Generalized Laplace prior", color = "tab:red", ax = axtm)

fignm.legend(loc = "upper right")
figtm.legend(loc = "upper right")

fignm.show()
figtm.show()


tracen, axn = plt.subplots()
tracet, axt = plt.subplots()
axn.plot(nm.beta[:, 0])
axt.plot(tm.beta[:, 0])
tracen.show()
tracet.show()
