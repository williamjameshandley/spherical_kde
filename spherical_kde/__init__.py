""" The spherical kernel density estimator class. """

import matplotlib
import numpy
import cartopy.crs
from scipy.special import logsumexp
from spherical_kde.utils import decra_from_polar, polar_from_decra
from spherical_kde.distributions import (VonMisesFisher_distribution as VMF,
                                         VonMises_std)


class SphericalKDE(object):
    """ Spherical kernel density estimator

    Parameters
    ----------
    phi_samples, theta_samples : array_like
        spherical-polar samples to construct the kde

    weights : array_like
        Sample weighting
        default [1] * len(phi_samples))

    bandwidth : float
        bandwidth of the KDE. Increasing bandwidth increases smoothness

    density : int
        number of grid points in theta and phi to draw contours.

    Attributes
    ----------
    phi, theta : numpy.array
        spherical polar samples

    weights : numpy.array
        Sample weighting (normalised to sum to 1).

    bandwidth : float
        Bandwidth of the kde. defaults to rule-of-thumb estimator
        https://en.wikipedia.org/wiki/Kernel_density_estimation
        Set to None to use this value

    density : int
        number of grid points in theta and phi to draw contours.

    palefactor : float
        getdist-style colouration factor of sigma-contours.
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

        sigmahat = VonMises_std(self.theta, self.phi)
        self.suggested_bandwidth = 1.06*sigmahat*len(weights)**-0.2

    def __call__(self, phi, theta):
        """ Log-probability density estimate

        Parameters
        ----------
        phi, theta : float or array_like
            Spherical polar coordinate

        Returns
        -------
        float or array_like
            log-probability area density
        """
        return logsumexp(VMF(phi, theta, self.phi, self.theta, self.bandwidth),
                         axis=-1, b=self.weights)

    def plot(self, ax, colour='g', **kwargs):
        """ Plot the KDE on an axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            matplotlib axis to plot on. This must be constructed with a
            `cartopy.crs.projection`:

            >>> import cartopy
            >>> import matplotlib.pyplot as plt
            >>> fig = plt.subplots()
            >>> ax = fig.add_subplot(111, projection=cartopy.crs.Mollweide())

        color
            Colour to plot the contours.
            *arg* can be an *RGB* or *RGBA* sequence or a string in any of
            several forms:
                1) a letter from the set 'rgbcmykw'
                2) a hex color string, like '#00FFFF'
                3) a standard name, like 'aqua'
                4) a string representation of a float, like '0.4',
            This is passed into `matplotlib.colors.colorConverter.to_rgb`

        Keywords
        --------
        Any other keywords are passed to `matplotlib.axes.Axes.contourf`
        """
        try:
            if not isinstance(ax.projection, cartopy.crs.Projection):
                raise TypeError("ax.projection must be type"
                                "cartopy.crs.Projection "
                                "({})".format(type(ax.projection)))
        except AttributeError:
            raise TypeError("ax must be set up with cartopy.crs.Projection")

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
        ax.contourf(X, Y, P, levels=levels, colors=self._colours(colour),
                    transform=cartopy.crs.PlateCarree(), *kwargs)

    def plot_samples(self, ax, nsamples=None, **kwargs):
        """ Plot equally weighted samples on an axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            matplotlib axis to plot on. This must be constructed with a
            `cartopy.crs.projection`:

            >>> import cartopy
            >>> import matplotlib.pyplot as plt
            >>> fig = plt.subplots()
            >>> ax = fig.add_subplot(111, projection=cartopy.crs.Mollweide())

        nsamples : int
            Approximate number of samples to plot. Can only thin down to
            this number, not bulk up

        Keywords
        --------
        Any other keywords are passed to `matplotlib.axes.Axes.plot`

        """
        ra, dec = self._samples(nsamples)
        ax.plot(ra, dec, 'k.', transform=cartopy.crs.PlateCarree(), *kwargs)

    @property
    def bandwidth(self):
        if self._bandwidth is None:
            return self.suggested_bandwidth
        else:
            return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value

    def _samples(self, nsamples=None):
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
