import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
import particles
from particles import smc_samplers as ssp
from particles import distributions as dists
from particles import resampling as rs

class TemperedImportance :

    def __init__(self, X, Y, alpha = 1, nugget = None, sig2 = None,
            lamb = None, ksig2 = 1, thetasig2 = 1, bridge = False,
            klambda = 2, thetalambda = 2, step = 1e-2, plot = False) :

        self.X_ = X
        self.y_ = Y
        self.alpha = 1 if not bridge else alpha
        self.nugget = nugget
        self.sig2 = sig2
        self.lamb = lamb
        self.ksig2 = ksig2
        self.thetasig2 = thetasig2
        self.klambda = klambda
        self.thetalambda = thetalambda
        self.bridge = bridge
        self.sig2_known = sig2 != None
        self.n, self.p = X.shape
        self.step = step
        self.plot = plot

        if self.nugget == None and np.all(np.linalg.eigvals(X.T.dot(X)) > 0) :
            self.nugget = 0
        elif self.nugget == None :
            nugmin = -4
            while not np.all(np.linalg.eigvals(X.T.dot(X)+ 10**nugmin *np.eye(self.p)) > 0) :
                nugmin += 1
            self.nugget = 10**nugmin
            print("Matrix is not positive definite, nugget set to :", self.nugget)
        self.C = np.linalg.cholesky(X.T.dot(X) + self.nugget * np.eye(self.p))
        Cmean = scipy.linalg.solve_triangular(self.C, X.T.dot(Y), lower = True)
        self.mean = scipy.linalg.solve_triangular(self.C.T, Cmean, lower = False)
        if not self.sig2_known : 
            self.Yz = Y.T.dot(Y) - Cmean.T.dot(Cmean)

    @property
    def base_dist(self) :

        if self.sig2_known :
            dist = {'beta' : mv_gaussian(mean = self.mean, C = self.C, sig2 = self.sig2)}
            base_dist = dists.StructDist(dist)
        else :
            chain_dist = dists.OrderedDict()
            a_ = 0.5*(self.n-self.p) if self.bridge else 0.5*self.n
            chain_dist['logsig2'] = dists.LogD(dists.InvGamma(a = self.ksig2 + a_, b = self.thetasig2 + 0.5*self.Yz))
            chain_dist['beta'] = dists.Cond(lambda v : mv_gaussian(mean = self.mean, C = self.C, sig2 = np.exp(v['logsig2'])),
                    dim = self.C.shape[0])
            base_dist = dists.StructDist(chain_dist)

        return base_dist

    def run(self, size = None, len_chain = 10) :

        imp = importance_model(base_dist = self.base_dist, p = self.p, sig2 = self.sig2, alpha = self.alpha,
                bridge = self.bridge, nugget = self.nugget)
        fk_tempering = LassoAdaptiveTempering(imp, len_chain = len_chain, wastefree = True, lambtemp = self.lamb,
                klambda = self.klambda, thetalambda = self.thetalambda, step = self.step, plot = self.plot)
        self.temp_alg = particles.SMC(fk = fk_tempering, N = int(size/len_chain), ESSrmin = 1.,
                verbose = True)
        self.temp_alg.run()
        self.W = self.temp_alg.wgts.W
        theta = self.temp_alg.X.theta
        if not self.sig2_known :
            self.v = np.empty(size)
        self.beta = np.empty((size, self.p))
        for i in range(size):
            if not self.sig2_known :
                s, b = theta[i]
                self.v[i] = np.exp(s)
                self.beta[i, :] = b
            else :
                self.beta[i, :] = theta[i]

        return self



class importance_model(ssp.TemperingBridge) :

    def __init__(self, base_dist, p, sig2 = None, alpha = 1, bridge = False, nugget = 0) :

        super().__init__(base_dist = base_dist)
        self.prior = base_dist
        self.p = p
        self.alpha = alpha if bridge else 1
        self.bridge = bridge
        self.nugget = nugget
        self.sig2 = sig2
        self.sig2_known = sig2 is not None

    def logtarget(self, theta) :
        logt = np.empty(theta['beta'].shape[0])
        for i in range(theta['beta'].shape[0]) :
            sig_ = self.sig2 if self.sig2_known else np.exp(theta['logsig2'][i])
            b_ = 1 if self.bridge else np.sqrt(sig_)
            logt[i] = (-np.sum(np.abs(theta['beta'][i, :])**self.alpha)/b_)
        logt += self.prior.logpdf(theta)
        return logt


class mv_gaussian(dists.ProbDist) :

    def __init__(self, mean, C, sig2) :

        self.sig2 = sig2
        self.sig2_known = (type(sig2) == int or type(sig2) == float)
        self.mean = mean
        self.C = C
        self.logdet = np.sum(np.log(np.diag(self.C)))

    def rvs(self, size = None) :

        beta = np.empty((size, self.dim))
        for i in range(size) :
            sig_ = self.sig2 if self.sig2_known else self.sig2[i]
            z = np.random.randn(self.dim)
            Cvar = scipy.linalg.solve_triangular(self.C.T, z, lower = False)
            beta[i, :] = (self.mean + np.sqrt(sig_)*Cvar)
        return beta

    def logpdf(self, x) :
        logpdf = np.empty(x.shape[0])
        for i in range(x.shape[0]) :
            sig_ = self.sig2 if self.sig2_known else self.sig2[i]
            root = self.C.T.dot(x[i, :] - self.mean)
            logpdf[i] = (-0.5/sig_ *root.T.dot(root) - 0.5*self.dim*np.log(2*np.pi*sig_) + self.logdet)
        return logpdf

    @property
    def dim(self) :
        return self.C.shape[0]


class LassoAdaptiveTempering(ssp.AdaptiveTempering) :

    def __init__(self, model = None, wastefree = True, len_chain = 10, move = None,
        ESSrmin = 0.5, lambtemp = None, klambda = 2, thetalambda = 2,
        step = 1e-2, plot = False) :

        super().__init__(model = model, wastefree = wastefree, len_chain = len_chain,
            move = move, ESSrmin = ESSrmin)
        self.lambtemp = lambtemp
        self.klambda = klambda
        self.thetalambda = thetalambda
        self.step = step
        self.plot = plot

    def lambc(self, x) :
        if not self.lambtemp is None :
            return self.lambtemp
        else :
            numax = None
            logp = lambda nu : ((self.model.p/self.model.alpha + self.klambda - 1)*np.log(nu)- nu*(self.thetalambda - x.llik))
            lpnu = lambda nu : rs.log_mean_exp(logp(nu))
            nup = [self.step, 2*self.step]
            lpnup = [lpnu(nup[0]), lpnu(nup[1])]
            loga  = rs.log_sum_exp_ab(lpnup[0], lpnup[1])
            while numax is None :
                #a = np.trapz(y = np.exp(lpnup), x = nup) #to use the trapezoidal rule
                newnu = nup[-1] + self.step
                newlpnu = lpnu(newnu)
                if newlpnu < np.log(1e-3) + loga + np.log(self.step) :
                    numax = newnu
                else :
                    loga = rs.log_sum_exp_ab(loga, newlpnu) #using rectangular integration from log values
                    nup.append(newnu)
                    lpnup.append(newlpnu)
            print("Penalization set to max value :", numax, "according to posterior p(lamb | Y)")
            if self.plot : 
                ess = lambda nu  : rs.essl(logp(nu))
                essp = [ess(nu) for nu in nup]
                sns.set_theme()
                fig, ax = plt.subplots()
                ax.plot(nup, rs.exp_and_normalise(np.log(self.step) + lpnup), color = "tab:blue")
                axess = ax.twinx()
                axess.plot(nup, np.array(essp)/x.N, color = "tab:orange")
                plt.show()
            return numax


    def done(self, smc) :
        if smc.X is None :
            return False #We have not started yet
        else : 
            return smc.X.shared['exponents'][-1] >= self.lamb

    def logG(self, t, xp, x) :
        if t == 0 :
            self.lamb = self.lambc(x)
        ESSmin = self.ESSrmin * x.N
        f = lambda e : rs.essl(e * x.llik) - ESSmin
        epn = x.shared['exponents'][-1]
        if f(self.lamb - epn) > 0 : #we're done (last iteration)
            delta = self.lamb - epn
            new_epn = self.lamb
        else : 
            delta = scipy.optimize.brentq(f, 1e-12, self.lamb - epn) #secant search
            #left endpoint is > 0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        x.shared['exponents'].append(new_epn)
        return self.logG_tempering(x, delta, wnugget = True) if t == 1 else self.logG_tempering(x, delta)

    def logG_tempering(self, x, delta, wnugget = False):
    
        dl = delta * x.llik
        if wnugget :
            nuglik = np.empty(x.theta.size)
            for i in range(x.theta.size) :
                if not self.model.sig2_known :
                    logs, b = x.theta[i]
                    s = np.exp(logs)
                else :
                    s = self.model.sig2
                    b = x.theta[i]
                nuglik[i] = 0.5*self.model.nugget/s *np.linalg.norm(b, ord = 2)
            dl += nuglik
        x.lpost += dl
        self.update_path_sampling_est(x, delta)
        return dl

