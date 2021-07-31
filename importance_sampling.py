import numpy as np
import scipy

class importance :

    def __init__(self, N = 1000, alpha = 1, nugget = 0, sig2 = None, lamb  = None, 
            ksig2 = 1, thetasig2 = 1, klambda = 2, thetalambda = 2, bridge = False):

        self.N = N
        self.alpha = 1 if not bridge else alpha 
        self.sig2 = [sig2 for k in range(N)]
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
        C = np.linalg.cholesky(X.T.dot(X)+self.nugget*np.eye(self.p))
        Cmean = scipy.linalg.solve_triangular(C, X.T.dot(Y), lower = True)
        if not self.sig2_known :
            Yz = Y.T.dot(Y) - Cmean.T.dot(Cmean)
            a = self.ksig2 + 0.5*(n-p) if self.bridge else self.ksig2+0.5*n
            self.sig2 = invgamma.rvs(a = a, scale = self.thetasig2+0.5*Yz, size = self.N)
        self.beta = np.empty((self.N, self.p))
        mean = scipy.linalg.solve_triangular(C.T, Cmean, lower = False)
        for i in range(self.N):
            z = np.random.randn(self.p)
            Cvar = scipy.linalg.solve_triangular(C.T, z, lower = False)
            self.beta[i, :] = (mean+np.sqrt(self.sig2[i])*Cvar)
        if not self.lamb_known :
            self.lamb = np.empty(self.N)
            b_ = 1 if self.bridge else 1/self.sig2  
            for i in range(self.N): 
                self.lamb[i] = gamma.rvs(a = self.klambda + self.p/self.alpha,
                        scale = 1/(self.thetalambda+b_[i]*np.sum(np.abs(beta[i, :])**self.alpha)), size = 1)
            self.tau = self.lamb**(-1/self.alpha)
        return(self)

    def weight(self):
        
        lw = np.empty(self.N)
        for i in range(self.N):
            zi = np.abs(self.beta[i, :])
            bi_ = 1 if self.bridge else self.sig2[i]
            if self.lamb_known :
                lw[i] = -self.lamb * np.sum(zi**self.alpha)/bi_ + self.nugget * np.linalg.norm(self.beta[i, :], ord = 2)/self.sig2[i]
            else :
                lw[i] = -(self.p/self.alpha+self.klambda)*np.log(self.thetalambda+np.sum(zi)/bi_) + self.nugget*np.linalg.norm(self.beta[i, :], ord = 2)/self.sig2[i]
        self.w = np.exp(lw - np.max(lw))
        self.w /= np.sum(self.w)
        self.ess = np.sum(self.w)**2 / np.sum(self.w**2)
        return(self)

    def fit(self, X, Y):

        self.mv_gaussian(X, Y).weight()
        mapb = np.empty(self.p)
        for i in range(self.p):
            mapb[i] = np.average(self.beta[:, i], weights = self.w)
        self.coef_ = mapb
        return(self)

    def predict(self, X):

        return(X.dot(self.coef_))

    def get_params(self, deep = True):

        return{'N' : self.N, 'alpha' : self.alpha, 'nugget' : self.nugget, 'sig2' : self.sig2, 
                'lamb' : self.lamb, 'ksig2' : self.ksig2, 'thetasig2' : self.thetasig2, 
                'klambda' : self.klambda, 'thetalambda' : self.thetalambda,
                'bridge' : self.bridge}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
