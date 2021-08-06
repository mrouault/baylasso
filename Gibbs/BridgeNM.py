import numpy as np
import scipy
from scipy.stats import gamma, invgamma
import TiltedStable

class bridgenm :

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
        self.lamb = np.empty((N, self.p))
        if not self.sig2_known :
            self.sig2 = np.empty(N)
            self.sig2[0] = invgamma.rvs(a = self.ksig2, scale = self.thetasig2, size = 1)
        if not self.nu_known :
            self.nu = np.empty(N)
            self.nu[0] = gamma.rvs(a = self.klambda, scale = 1/self.thetalambda, size = 1)
        self.tau = self.nu**(-1/self.alpha) if self.nu_known else self.nu[0]**(-1/self.alpha)
        self.ytemp = self.y_
        self.Vinv = self.X_.T.dot(self.X_)
        C = np.linalg.cholesky(self.Vinv)
        Cmean = scipy.linalg.solve_triangular(C, self.X_.T.dot(self.y_), lower = True)
        mu = scipy.linalg.solve_triangular(C.T, Cmean , lower = False)
        self.beta[0, :] = mu + np.random.randn(self.p)

        for i in range(1, N):
            self.lamb[i, :] = self.update_lamb(i)
            self.beta[i, :] =  self.update_beta(i)
            if not self.sig2_known :
                self.sig2[i] = self.update_sig2(i)
            if not self.nu_known :
                self.nu[i] = self.update_nu(i)
            if self.debug :
                self.ytemp = self.update_y(i)

        return self
            
    def update_beta(self, i):

        L = np.diag(self.lamb[i, :])
        sig_ = self.sig2 if self.sig2_known else self.sig2[i-1]
        C = np.linalg.cholesky(self.Vinv + 2*sig_/self.tau**2 * L)
        Cmean = scipy.linalg.solve_triangular(C, self.X_.T.dot(self.ytemp), lower = True)
        mean = scipy.linalg.solve_triangular(C.T, Cmean, lower = False)
        z = np.random.randn(self.p)
        Cvar = scipy.linalg.solve_triangular(C.T, z, lower = False)
        return mean + np.sqrt(sig_)*Cvar

    def update_lamb(self, i):

        lamb = np.empty(self.p)
        x = abs(self.beta[i-1, :]/self.tau)**2
        for j in range(self.p):
            lamb[j] = 2*TiltedStable.tilted_stable(0.5*self.alpha, x[j])
        return lamb

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
        return ytemp

