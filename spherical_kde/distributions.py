import numpy
from scipy.special import ive as ModifiedBessel, gamma as Gamma
from spherical_kde.utils import cartesian_from_spherical, infinite_sum

def logsinh(x):
   return x + numpy.log(1-numpy.exp(-2*x)) - numpy.log(2)

def VMF(phi, theta, phi0, theta0, sigma):
    """ https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution """
    x = cartesian_from_spherical(phi, theta)
    x0 = cartesian_from_spherical(phi0, theta0)
    return -numpy.log(4*numpy.pi*sigma**2) - logsinh(1./sigma**2) + numpy.tensordot(x,x0,axes=[[0],[0]])/sigma**2


class KentDistribution(object):
    def __init__(self):
        self._kappa = None
        self._beta = None

    def c(self, kappa, beta):
        if kappa != self._kappa or beta != self._beta:
            self._kappa = kappa
            self._beta = beta
            self._c = infinite_sum( lambda j: Gamma(j+1./2)/Gamma(j+1) * beta**(2*j) * (kappa/2)**(-2*j-1./2)*ModifiedBessel(2*j+1./2,kappa) )
        return self._c

    def __call__(self, theta, phi, kappa, beta, g1, g2, g3):
        if kappa < 0:
            raise ValueError("KentDistribution: Parameter kappa ({}) must be >= 0".format(kappa))
        elif beta < 0:
            raise ValueError("KentDistribution: Parameter beta ({}) must be >= 0".format(beta))
        elif 2*beta > kappa:
            raise ValueError("KentDistribution: Parameter beta ({}) must be <= kappa/2 ({})".format(beta, kappa/2))


        x = cartesian_from_spherical(theta, phi)
        gx1 = x.transpose().dot(g1).transpose()
        gx2 = x.transpose().dot(g2).transpose()
        gx3 = x.transpose().dot(g3).transpose()
        return - numpy.log(self.c(kappa, beta)) + kappa * gx1 + beta * (gx2**2 - gx3**2)
