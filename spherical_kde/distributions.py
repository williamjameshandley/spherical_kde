import numpy
from spherical_kde.utils import cartesian_from_spherical, logsinh


def VonMisesFisherDistribution(phi, theta, phi0, theta0, sigma):
    """ Von-Mises Fisher distribution function.

    https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution

    Args:
        phi, theta (float or array-like)
            Ppherical-polar coordinates to evaluate function at.

        phi0, theta0 (float or array-like)
            Center of the distribution.

        sigma
            Width of the distribution.
    """
    x = cartesian_from_spherical(phi, theta)
    x0 = cartesian_from_spherical(phi0, theta0)
    return (-numpy.log(4*numpy.pi*sigma**2) - logsinh(1./sigma**2)
            + numpy.tensordot(x, x0, axes=[[0], [0]])/sigma**2)
