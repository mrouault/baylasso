import numpy as np
import scipy
from scipy.stats import gamma, truncnorm, invgamma

class bridgetm :

    def __init__(self, X, Y, sig2 = None, nu = None, alpha = 1,
            ksig2 = 1, thetasig2 = 1, klambda = 2, thetalambda = 2, debug = False):

        self.X_ = X
        self.y_ = Y
        self.n, self.p = X.shape
        self.sig2 = sig2
        self.nu = nu
        self.alpha = alpha
        self.ksig2 = ksig2
        self.thetasig2 = thetasig2
        self.klambda = klambda
        self.thetalambda = thetalambda
        self.sig2_known = sig2 != None
        self.nu_known = nu != None
        self.debug = debug

    def run(self, N):

        self.N = N
        self.beta = np.empty((N, self.p))
        self.u = np.empty((N, self.p))
        self.W = np.empty((N, self.p))
        if not self.sig2_known :
            self.sig2 = np.empty(N)
            self.sig2[0] = invgamma.rvs(a = self.ksig2, scale = self.thetasig2, size = 1)
        if not self.nu_known :
            self.nu = np.empty(N)
            self.nu[0] = gamma.rvs(a = self.klambda, scale = 1/self.thetalambda, size = 1)
        self.tau = self.nu**(-1/self.alpha) if self.nu_known else self.nu[0]**(-1/self.alpha)
        self.ytemp = self.y_
        self.Vinv = self.X_.T.dot(self.X_)
        self.C = np.linalg.cholesky(self.Vinv)
        Cmean = scipy.linalg.solve_triangular(self.C, self.X_.T.dot(self.ytemp), lower = True)
        self.mu = scipy.linalg.solve_triangular(self.C.T, Cmean, lower = False)
        self.beta[0, :] = self.mu + np.random.randn(self.p)
        self.W[0, :] = np.abs(self.beta[0, :])

        for i in range(1, self.N):
            self.u[i, :] = self.update_u(i)
            self.W[i, :] = self.update_w(i)
            self.beta[i, :] = self.update_beta(i)
            if not self.sig2_known :
                self.sig2[i] = self.update_sig2(i)
            if not self.nu_known :
                self.nu[i] = self.update_nu(i)
            if self.debug :
                self.ytemp = self.update_y(i)

        return self

    def update_beta(self, i):
    
        beta = np.empty(self.p)
        for j in range(self.p):
            j_ = np.array([k for k in range(self.p) if k !=j])
            betaj_ = [self.beta[i-1, k] if k > j else beta[k] for k in j_]
            bj = self.W[i, j]**(1/self.alpha) * self.tau * (1-self.u[i, j])
            meanj = self.mu[j] - self.Vinv[j, j_].dot(betaj_ - self.mu[j_])/self.Vinv[j, j]
            sig_ = self.sig2 if self.sig2_known else self.sig2[i-1]
            scalej = sig_/self.Vinv[j, j]
            aj = (-bj-meanj)/np.sqrt(scalej)
            cj = (bj-meanj)/np.sqrt(scalej)
            beta[j] = truncnorm.rvs(a = min(aj, cj), b = max(aj, cj), loc = meanj, scale = np.sqrt(scalej), size = 1)
        return beta

    def update_u(self, i):

        u = np.empty(self.p)
        for j in range(self.p):
            cj = 1- self.W[i-1, j]**(-1/self.alpha) *np.abs(self.beta[i-1, j]/self.tau)
            u[j] = np.random.random()*cj if cj >=0 else 0
        return u

    def update_w(self, i):
        
        w = np.empty(self.p)
        for j in range(self.p):
            aj = (abs(self.beta[i-1, j]/self.tau)/(1-self.u[i, j]))**self.alpha
            zj = np.random.random()
            if zj <= self.alpha/(1+self.alpha*aj):
                wj = gamma.rvs(a = 2, scale = 1)
            else :
                wj = gamma.rvs(a = 1, scale = 2)
            w[j] = wj + aj
        return w

    def update_sig2(self, i):

        a = self.ksig2 + 0.5*self.n
        dist = np.linalg.norm(self.ytemp - self.X_.dot(np.array(self.beta[i, :])), ord = 2)**2
        return invgamma.rvs(a = a, scale = self.thetasig2 + dist/2, size = 1)

    def update_nu(self, i):

        a = self.klambda + self.p/self.alpha
        rate = self.thetalambda + np.sum(np.abs(self.beta[i, :])**self.alpha)
        nu = gamma.rvs(size = 1, a = a, scale = 1/rate)
        self.tau = nu**(-1/self.alpha)
        return nu
    
    def update_y(self, i):

        sig_ = self.sig2 if self.sig2_known else self.sig2[i]
        ytemp = self.X_.dot(self.beta[i, :])+np.sqrt(sig_)*np.random.randn(self.n)
        Cmean = scipy.linalg.solve_triangular(self.C, self.X_.T.dot(ytemp), lower = True)
        self.mu = scipy.linalg.solve_triangular(self.C.T, Cmean, lower = False)
        return ytemp
