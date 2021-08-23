import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
import pandas as pd
import warnings
warnings.simplefilter('ignore')

from Gibbs.GibbsLasso import gibbslasso

sns.set_style('darkgrid')

X, Y = load_boston(return_X_y = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = (Y-np.mean(Y))/np.std(Y)
n, p = X.shape

C = np.linalg.cholesky(X.T.dot(X))
Cmean = scipy.linalg.solve_triangular(C, X.T.dot(Y), lower = True)
mean = scipy.linalg.solve_triangular(C.T, Cmean, lower = False)
sig2 = 1/n * np.sum((Y-X.dot(mean))**2)

N = 1000
lamb = 1

beta = gibbslasso(X = X, Y = Y, lamb = lamb, sig2 = sig2)
betasig2 = gibbslasso(X = X, Y = Y, lamb = lamb)

beta.run(N)
betasig2.run(N)

fig, ax = plt.subplots()
figsig2, axsig2 = plt.subplots()

pd.DataFrame(acf(beta.beta[100:, 0], fft = True)).plot(kind = "bar", ax = ax)
pd.DataFrame(acf(betasig2.beta[100:, 0], fft = True)).plot(kind = "bar", ax = axsig2)

fig.show()
figsig2.show()
