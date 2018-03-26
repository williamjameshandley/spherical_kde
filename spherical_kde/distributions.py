import numpy
import scipy.optimize
from spherical_kde.utils import (cartesian_from_polar,
                                 polar_from_cartesian, logsinh,
                                 rotation_matrix)


def VonMisesFisher_distribution(phi, theta, phi0, theta0, sigma):
    """ Von-Mises Fisher distribution function.

    https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution

    Args:
        phi, theta (float or array-like)
            Spherical-polar coordinates to evaluate function at.

        phi0, theta0 (float or array-like)
            Center of the distribution.

        sigma
            Width of the distribution.
    """
    x = cartesian_from_polar(phi, theta)
    x0 = cartesian_from_polar(phi0, theta0)
    return (-numpy.log(4*numpy.pi*sigma**2) - logsinh(1./sigma**2)
            + numpy.tensordot(x, x0, axes=[[0], [0]])/sigma**2)


def VonMisesFisher_sample(phi0, theta0, sigma, size=None):
    """ Draw a sample from the Von-Mises Fisher distribution.

    Args:
        phi0, theta0 (float or array-like)
            Center of the distribution.

        sigma
            Width of the distribution.

        size (int, tuple, array-like) default None
            number of samples to draw

    """
    n0 = cartesian_from_polar(phi0, theta0)
    M = rotation_matrix([0,0,1],n0)

    x = numpy.random.uniform(size=size)
    phi = numpy.random.uniform(size=size) * 2*numpy.pi
    theta = numpy.arccos(1 + sigma**2 * numpy.log(1 + (numpy.exp(-2/sigma**2)-1) * x))
    n = cartesian_from_polar(phi, theta) 

    x = M.dot(n)
    phi, theta = polar_from_cartesian(x)

    return phi, theta


def VonMises_mean(phi, theta):
    """ Von-Mises sample mean.

    As advised in

    https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters

    Args:
        phi, theta (array-like)
            Spherical-polar coordinate samples to compute mean from.

    Returns:
        \sum_i^N x_i / || \sum_i^N x_i ||
    """
    x = cartesian_from_polar(phi, theta)
    S = numpy.sum(x, axis=-1)
    phi, theta = polar_from_cartesian(S)
    return phi, theta


def VonMises_standarddeviation(phi, theta):
    """ Von-Mises sample standard deviation.

    As advised in

    https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters

    but re-parameterised for sigma rather than kappa

    Args:
        phi, theta (array-like)
            Spherical-polar coordinate samples to compute mean from.

    Returns:
        solution for 1/tanh(x) - 1/x = R,
        where R = || \sum_i^N x_i || / N
    """
    x = cartesian_from_polar(phi, theta)
    S = numpy.sum(x, axis=-1)
    R = S.dot(S)**0.5/x.shape[-1]

    def f(s):
        return 1/numpy.tanh(s)-1./s-R

    kappa = scipy.optimize.brentq(f, 1e-8, 1e8)
    sigma = kappa**-0.5
    return sigma
