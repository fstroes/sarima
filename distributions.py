# Distributions needed for the non-Gaussian SARIMA
import numpy as np
from scipy.special import gamma, gammaln

class student_t:
    def __init__(self, nu=20, mu=0, sigma=1):
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
    # Gamma[(ν + 1) / 2] / (σ Sqrt[ν Pi] Gamma[ν / 2]) / ((ν + ((x - μ) / σ) ^ 2) / ν) ^ ((ν + 1) / 2)
    def pdf(self, x, nu=None, mu=None, sigma=None):
        if nu == None:
            nu=self.nu
        if mu == None:
            mu = self.mu
        if sigma == None:
            sigma = self.sigma

        density = gamma((nu + 1) / 2) / (sigma * np.sqrt(nu * np.pi) * gamma(nu / 2)) / ((nu + ((x - mu) / sigma) ** 2) / nu) ** ((nu + 1) / 2)
        return density


class ZINB:
    def __init__(self, pi =.5, alpha=1, mu=10):
        # 0=<pi<=1, 0<alpha, 0<mu
        self.pi = pi
        self.alpha = alpha
        self.mu = mu

    def pdf(self, x, pi=None, alpha=None, mu=None):
        if pi == None:
            pi = self.pi
        if alpha == None:
            alpha = self.alpha
        if mu == None:
            mu = self.mu

    def log_L(self, x, pi=None, alpha=None, mu=None):
        if pi == None:
            pi = self.pi
        if alpha == None:
            alpha = self.alpha
        if mu == None:
            mu = self.mu

        if mu == np.inf:
            return -1e9
        else:
            a_inv = alpha**(-1)

            if x == 0:
                A = (1 - pi)
                B = (a_inv/(a_inv+mu) )
                loglik = np.log(pi + A * (B ** (a_inv)) )

            else:
                A = gammaln(x+a_inv) - gammaln(x+1) - gammaln(a_inv) # a complicated term in the log likelihood function
                B = np.log( a_inv / (a_inv + mu) )
                C = np.log( mu / (a_inv + mu))

                loglik = np.log(1 - pi) + A + a_inv * B + x * C

            if np.isnan(loglik):
                return -1e9
            if np.inf == loglik:
                return -1e9
            else:
                return loglik

    def score(self, x , pi=None, alpha=None, mu=None):
        # The score (derivative of log-likelihood with respect to parameter for given value)
        if pi == None:
            pi = self.pi
        if alpha == None:
            alpha = self.alpha
        if mu == None:
            mu = self.mu
        return ((x)-mu)/(alpha * mu+1) if x > 0 else ((pi-1) * mu)/((alpha * mu + 1)*(1+pi * (alpha * mu+1)**(alpha-1)-pi)) if x == 0 else None
