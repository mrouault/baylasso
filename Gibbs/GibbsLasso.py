import numpy as np
import scipy
from scipy.stats import invgauss, invgamma, gamma

class gibbslasso :

    def __init__(self, X, Y, sig2 = None, lamb = None,
            ksig2 = 1, thetasig2 = 1, klambda = 2, thetalambda = 2, debug = False):
        
        self.X_ = X
        self.y_ = Y
        self.n, self.p = X.shape
        self.sig2 = sig2
        self.lamb = lamb
        self.ksig2 = ksig2
        self.thetasig2 = thetasig2
        self.klambda = klambda
        self.thetalambda = thetalambda
        self.sig2_known = sig2 != None
        self.lamb_known = lamb != None
        self.debug = debug
    
    def run(self, N):

        self.N = N
        self.tau2inv = np.empty((self.N, self.p))
        self.beta = np.empty((self.N, self.p))
        if not self.sig2_known :
            self.sig2 = np.empty(self.N)
            self.sig2[0] = invgamma.rvs(size = 1, a = self.ksig2, scale = self.thetasig2)
        if not self.lamb_known :
            self.lamb = np.empty(self.N)
            self.lamb[0] = gamma.rvs(size = 1, a = self.klambda, scale = 1/self.thetalambda)
        self.ytemp = self.y_

        lamb_ = self.lamb if self.lamb_known else self.lamb[0]
        tau2 = gamma.rvs(size = self.p, a = 1, scale = 2/lamb_**2)
        self.tau2inv[0] = 1/tau2
        self.Vinv = self.X_.T.dot(self.X_)
        self.xy = self.X_.T.dot(self.ytemp)

        for i in range(1, self.N):
            self.beta[i, :] = self.update_beta(i)
            self.tau2inv[i, :] = self.update_tau2inv(i)
            if not self.sig2_known : 
                self.sig2[i] = self.update_sig2(i)
            if not self.lamb_known :
                self.lamb[i] = self.update_lamb(i)
            if self.debug :
                self.ytemp = self.update_y(i)

        return self


    def update_beta(self, i):
    
        D_tau2inv = np.diag(self.tau2inv[i-1, :])
        L = np.linalg.cholesky(self.Vinv + D_tau2inv)
        Lc = scipy.linalg.solve_triangular(L, self.xy, lower = True)
        mean = scipy.linalg.solve_triangular(L.T, Lc, lower = False)
        z = np.random.randn(self.p)
        Lvar = scipy.linalg.solve_triangular(L.T, z, lower = False)
        sig_ = self.sig2 if self.sig2_known else self.sig2[i-1]
        return mean + np.sqrt(sig_)*Lvar

    def update_tau2inv(self, i):

        lamb_ = self.lamb if self.lamb_known else self.lamb[i-1]
        sig_ = self.sig2 if self.sig2_known else self.sig2[i-1]
        beta = self.beta[i, :]
        mu = [np.sqrt(lamb_**2 * sig_ / beta[j]**2) for j in range(self.p)]
        return [invgauss.rvs(mu[j]/lamb_**2, scale = lamb_**2, size = 1) for j in range(self.p)]

    def update_sig2(self, i):

        a = 0.5*(self.n+self.p)+self.ksig2
        beta = self.beta[i, :]
        tau2inv = self.tau2inv[i, :]
        beta_Dtau = sum([beta[j]**2 * tau2inv[j] for j in range(self.p)])
        v = np.linalg.norm(self.ytemp - self.X_.dot(np.array(beta)), ord = 2)**2
        scale = self.thetasig2 + 0.5*(beta_Dtau + v)
        return invgamma.rvs(a = a, scale = scale, size = 1)

    def update_lamb(self, i):

        a = self.klambda + self.p
        sig_ = self.sig2 if self.sig2_known else self.sig2[i-1]
        beta = self.beta[i, :]
        rate = self.thetalambda + np.sum(np.abs(beta))/sig_
        return gamma.rvs(a = a, scale = 1/rate, size = 1)

    def update_y(self, i):
    
        sig_ = self.sig2 if self.sig2_known else self.sig2[i]
        ytemp = self.X_.dot(self.beta[i, :]) + np.sqrt(sig_)*np.random.randn(self.n)
        self.xy = self.X_.T.dot(ytemp)
        return ytemp
