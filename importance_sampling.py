import numpy as np
import scipy
from scipy.stats import invgamma, gamma

class importance :

    def __init__(self, N = 1000, alpha = 1, nugget = None, sig2 = None, lamb  = None, 
            ksig2 = 1, thetasig2 = 1, klambda = 2, thetalambda = 2, bridge = False):

        self.N = N
        self.alpha = 1 if not bridge else alpha 
        self.sig2 = sig2
        self.lamb = lamb
        self.nugget = nugget
        self.ksig2 = ksig2
        self.thetasig2 = thetasig2
        self.klambda = klambda
        self.thetalambda = thetalambda
        self.bridge = bridge
        self.sig2_known = sig2 != None
        self.lamb_known = lamb != None

    def mv_gaussian(self, X, Y):

        self.X_ = X
        self.y_ = Y
        self.n, self.p = X.shape
        if self.nugget == None and np.all(np.linalg.eigvals(X.T.dot(X)) > 0):
            self.nugget = 0
        elif self.nugget == None :
            nugmin = -5
            while not np.all(np.linalg.eigvals(X.T.dot(X)+10**nugmin *np.eye(self.p)) > 0):
                nugmin +=1
            self.nugget = 10**nugmin
        C = np.linalg.cholesky(X.T.dot(X)+self.nugget*np.eye(self.p))
        Cmean = scipy.linalg.solve_triangular(C, X.T.dot(Y), lower = True)
        if not self.sig2_known :
            Yz = Y.T.dot(Y) - Cmean.T.dot(Cmean)
            a = self.ksig2 + 0.5*(self.n-self.p) if self.bridge else self.ksig2+0.5*self.n
            self.sig2 = invgamma.rvs(a = a, scale = self.thetasig2+0.5*Yz, size = self.N)
        self.beta = np.empty((self.N, self.p))
        mean = scipy.linalg.solve_triangular(C.T, Cmean, lower = False)
        if not self.lamb_known :
            self.lamb = np.empty(self.N)
        for i in range(self.N):
            z = np.random.randn(self.p)
            Cvar = scipy.linalg.solve_triangular(C.T, z, lower = False)
            sig_ = self.sig2 if self.sig2_known else self.sig2[i]
            self.beta[i, :] = (mean+np.sqrt(sig_)*Cvar) 
            if not self.lamb_known : 
                b_ = 1 if self.bridge else 1/np.sqrt(sig_) 
                self.lamb[i] = gamma.rvs(a = self.klambda + self.p/self.alpha,
                        scale = 1/(self.thetalambda+b_*np.sum(np.abs(self.beta[i, :])**self.alpha)), size = 1)
        return(self)

    def weight(self):
        
        lw = np.empty(self.N)
        for i in range(self.N):
            zi = np.abs(self.beta[i, :])
            sig_ = self.sig2 if self.sig2_known else self.sig2[i]
            bi_ = 1 if self.bridge else np.sqrt(sig_)
            if self.lamb_known :
                lw[i] = -self.lamb * np.sum(zi**self.alpha)/bi_ + self.nugget * np.linalg.norm(self.beta[i, :], ord = 2)/sig_
            else :
                lw[i] = -(self.p/self.alpha+self.klambda)*np.log(self.thetalambda+np.sum(zi)/bi_) + self.nugget*np.linalg.norm(self.beta[i, :], ord = 2)/sig_
        self.w = np.exp(lw - np.max(lw))
        self.w /= np.sum(self.w)
        self.ess = np.sum(self.w)**2 / np.sum(self.w**2)
        return(self)

