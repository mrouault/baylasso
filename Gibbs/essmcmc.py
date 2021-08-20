import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
ro.r['source']('Gibbs/ress.R')
reffectivesize = ro.globalenv['leffectiveSize']


def essmcmc(beta) :
    nr, nc = beta.shape
    essmcmc = []
    for k in range(nc) :
        betark = ro.r.matrix(beta[:, k], nrow = nr, ncol = 1)
        ressk = reffectivesize(betark)
        essmcmc.append(ressk[0])
    return essmcmc
