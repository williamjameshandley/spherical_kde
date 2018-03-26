import matplotlib
from scipy.special import logsumexp
import numpy
from spherical_kde.distributions import (VonMisesFisher_distribution as VMF,
                                         VonMises_standarddeviation)
from spherical_kde.utils import decra_from_polar, polar_from_decra
#import cartopy.crs as ccrs


class SphericalKDE(object):
    """ Spherical kernel density estimator

    Args:
        phi_samples (array-like)
            azimuthal samples to construct the kde

        theta_samples (array-like)
            polar samples to construct the kde

        weights (array-like: default [1] * len(phi_samples))
            weighting for the samples

        bandwidth (float: default 0.2)
            bandwidth of the KDE. Increasing bandwidth increases smoothness

        density (int: default 100)
            number of grid points in theta and phi to draw contours.

    Attributes:
        phi (numpy.array)
            Azimuthal samples.

        theta (numpy.array)
            Polar samples.

        weights (numpy.array)
            Sample weighting (normalised to sum to 1).

        bandwidth (float)
            Bandwidth of the kde. defaults to rule-of-thumb estimator:
            https://en.wikipedia.org/wiki/Kernel_density_estimation
            Set to None to use this value

        density (int)
            number of gridpoints in each direction to evaluate KDE over sphere

        palefactor (float)
            Getdist-style colouration factor of sigma-contours.
    """
    def __init__(self, phi_samples, theta_samples,
                 weights=None, bandwidth=None, density=100):

        self.phi = numpy.array(phi_samples)
        self.theta = numpy.array(theta_samples)

        if weights is None:
            weights = numpy.ones_like(phi_samples)
        self.weights = numpy.array(weights) / sum(weights)
        self.bandwidth = bandwidth
        self.density = density
        self.palefactor = 0.6

        if len(self.phi) != len(self.theta):
            raise ValueError("phi_samples must be the same"
                             "shape as theta_samples ({}!={})".format(
                                 len(self.phi), len(self.theta)))
        if len(self.phi) != len(self.weights):
            raise ValueError("phi_samples must be the same"
                             "shape as weights ({}!={})".format(
                                 len(self.phi), len(self.weights)))

        sigmahat = VonMises_standarddeviation(self.theta, self.phi)
        self.suggested_bandwidth = 1.06*sigmahat*len(weights)**-0.2

    @property
    def bandwidth(self):
        if self._bandwidth is None:
            return self.suggested_bandwidth
        else:
            return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value

    def __call__(self, phi, theta):
        """ Log-probability density estimate

        Args:
            phi (float or array-like)
                azimuthal angle.

            theta (float or array-like)
                polar angle.
        """
        return logsumexp(VMF(phi, theta, self.phi, self.theta, self.bandwidth),
                         axis=-1, b=self.weights)

    def plot(self, ax, colour='g'):
        """ Plot the KDE on an axis. """
        #if not isinstance(ax.projection, ccrs.Projection):
        #    raise TypeError("ax.projection must be of type ccrs.Projection "
        #                    "({})".format(type(ax.projection)))

        # Compute the kernel density estimate on an equiangular grid
        ra = numpy.linspace(-180, 180, self.density)
        dec = numpy.linspace(-89, 89, self.density)
        X, Y = numpy.meshgrid(ra, dec)
        phi, theta = polar_from_decra(X, Y)
        P = numpy.exp(self(phi, theta))

        # Find 2- and 1-sigma contours
        Ps = numpy.exp(self(self.phi, self.theta))
        i = numpy.argsort(Ps)
        cdf = self.weights[i].cumsum()
        levels = [Ps[i[numpy.argmin(cdf < f)]] for f in [0.05, 0.33]]
        levels += [numpy.inf]

        # Plot the countours on a suitable equiangular projection
        ax.contourf(X, Y, P, levels=levels, colors=self._colours(colour))
                    #transform=ccrs.PlateCarree())

    def plot_decra_samples(self, ax, color='k', nsamples=None):
        """ Plot equally weighted samples on an axis. """
        ra, dec = self._decra_samples(nsamples)
        ax.plot(ra, dec, 'k.')#, transform=ccrs.PlateCarree())

    def _decra_samples(self, nsamples=None):
        weights = self.weights / self.weights.max()
        if nsamples is not None:
            weights /= weights.sum()
            weights *= nsamples
        i_ = weights > numpy.random.rand(len(weights))
        phi = self.phi[i_]
        theta = self.theta[i_]
        ra, dec = decra_from_polar(phi, theta)
        return ra, dec

    def _colours(self, colour):
        cols = [matplotlib.colors.colorConverter.to_rgb(colour)]
        for _ in range(1, 2):
            cols = [[c * (1 - self.palefactor) + self.palefactor
                     for c in cols[0]]] + cols
        return cols
