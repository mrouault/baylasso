import numpy as np
import scipy
from scipy.stats import gamma

def tilted_stable(alpha, x) :

    gam = alpha*(1-alpha) * x**alpha
    c1 = np.sqrt(np.pi/2)
    ksi = (2+c1)*np.sqrt(2*gam)/np.pi + 1
    psi = np.sqrt(gam*np.pi)*np.exp(-gam*np.pi**2 /8)*(2+c1)/np.pi
    w1 = ksi*c1/np.sqrt(gam)
    w2 = 2*psi*np.sqrt(np.pi)
    w3 = ksi*np.pi
    b = (1-alpha)/alpha
    
    flagX = False
    while not flagX :

        flagU = False
        while not flagU :
            U = gen_u(gam, w1, w2, w3)
            W = np.random.random()
            zeta = np.sqrt(zolotarev_b(U, alpha))
            phi = (np.sqrt(gam)+alpha*zeta)**(1/alpha)
            z = phi/(phi-np.sqrt(gam)**(1/alpha))
            c2 = 0
            if U >=0 and gam >= 1 :
                c2+= ksi*np.exp(-gam/2 * U**2)
            if 0 < U < np.pi :
                c2+= phi/np.sqrt(np.pi - U)
            if 0 <= U <= np.pi and gam < 1 :
                c2+= ksi
            rho = np.pi * np.exp(-x**alpha *(1-1/zeta**2))*c2
            rho/= (1+c1)*np.sqrt(gam)/zeta + z
            flagU = (U < np.pi and W*rho <= 1)

        a = zolotarev_a(U, alpha)
        m = (b*x /a)**alpha
        delta = np.sqrt(m*alpha/a)
        a1 = delta*c1
        a2 = delta
        a3 = z/a
        s = a1 + a2 + a3
        Vstar = np.random.random()
        Nstar = np.random.randn()
        Estar = gamma.rvs(a = 1, size = 1)
        if Vstar < a1/s :
            X = m - delta*np.abs(Nstar)
        elif Vstar < a2/s :
            X = m + delta*np.random.random()
        else :
            X = m + delta + a3*Estar
        E = -np.log(W*rho)
        flagX = X >=0 and condE(x, X, E, Estar, Nstar, a, m, delta, b)
    
    return X**(-b)

def zolotarev_a(x, alpha):

    return (np.sin(alpha*x)**alpha *np.sin((1-alpha)*x)**(1-alpha) /np.sin(x))**(1/(1-alpha))

def sincmm(x): 
    #instead of np.sinc, to ensure numeric stability and positive values,
    #see https://github.com/jwindle/BayesBridge/blob/master/Code/C/retstable.cpp
    ax = abs(x)
    if ax < 0.006 :
        if x == 0 :
            return 1
        if ax < 2e-4 :
            return 1-x**2 /6
        else :
            return 1- x**2 /6 *(1-x**2 /20)
    return np.sin(x)/x

def zolotarev_b(x, alpha):

    return (sincmm(x)/(sincmm(alpha*x)**alpha * sincmm((1-alpha)*x)**(1-alpha)))

def gen_u(gam, w1, w2, w3):

    V = np.random.random()
    Wstar = np.random.random()
    if gam >=1 and V < w1/(w1+w2):
        N = np.random.randn()
        U = np.abs(N)/np.sqrt(gam)
    elif gam >= 1:
        U = np.pi*(1-Wstar**2)
    elif V < w3/(w2 + w3) :
        U = np.pi*Wstar
    else :
        U = np.pi*(1-Wstar**2)
    return U

def condE(x, X, E, Estar, Nstar, a, m, delta, b):

    c3 = a*(X-m) + x*(X**(-b) - m**(-b))
    if X < m :
        c3+= -0.5 * Nstar**2
    if X > m + delta :
        c3+= -Estar
    return (c3 <= E)


