import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
from statsmodels.tsa.stattools import acf
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

N = 1000
alpha = 1.4
tm = bridgetm(X = X, Y = Y, alpha = alpha)
nm = bridgenm(X = X, Y = Y, alpha = alpha)

tm.run(N)
nm.run(N)

fig, ax = plt.subplots()
figacf, axacf = plt.subplots()
fig2, ax2 = plt.subplots()
figacf2, axacf2 = plt.subplots()

ax.plot(tm.beta[:, 0])
ax2.plot(nm.beta[:, 0])

pd.DataFrame(acf(tm.beta[100:, 0], fft = True)).plot(kind = "bar", ax = axacf)
pd.DataFrame(acf(nm.beta[100:, 0], fft = True)).plot(kind = "bar", ax = axacf2)

figh, axh = plt.subplots()
sns.histplot(x = tm.beta[:, 0], stat = "density", kde = True, ax = axh,
        label = "Triangular mixture", color = "tab:blue")
sns.histplot(x = nm.beta[:, 0], stat = "density", kde = True, ax = axh,
        label = "Normal mixture", color = "tab:orange")
figh.legend(loc = "upper right")

fig.show()
figacf.show()
fig2.show()
figacf2.show()
figh.show()
