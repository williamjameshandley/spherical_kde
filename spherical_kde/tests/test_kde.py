import spherical_kde
import numpy
import pytest
import cartopy.crs as ccrs
from numpy.testing import assert_allclose
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from spherical_kde.tests.test_distributions import random_phi_theta_sigma
from spherical_kde.utils import spherical_integrate, spherical_kullback_liebler
from spherical_kde.distributions import (VonMisesFisher_sample,
                                         VonMisesFisher_distribution)


def random_kde(nsamples):
    phi0, theta0, sigma0 = random_phi_theta_sigma()
    phi, theta = VonMisesFisher_sample(phi0, theta0, sigma0, size=nsamples)
    kde = spherical_kde.SphericalKDE(phi, theta)
    return kde, phi0, theta0, sigma0


def test_kde_lengths():
    numpy.random.seed(seed=0)
    kde, phi0, theta0, sigma0 = random_kde(100)
    assert len(kde.phi) == len(kde.theta)
    assert len(kde.phi) == len(kde.weights)


def test_kde_incorrect_lengths():
    with pytest.raises(ValueError):
        spherical_kde.SphericalKDE([1, 2], [0])
    with pytest.raises(ValueError):
        spherical_kde.SphericalKDE([1, 2], [1, 2], [1])


def test_kde_bandwith_automatic():
    numpy.random.seed(seed=0)
    kde, phi0, theta0, sigma0 = random_kde(100)
    assert kde.bandwidth > 0
    assert kde._bandwidth is None
    assert kde.bandwidth == kde.suggested_bandwidth
    kde.bandwidth = 5.
    assert kde._bandwidth == 5.
    assert kde.bandwidth == 5.


def test_kde_plotting():
    numpy.random.seed(seed=0)
    kde = random_kde(100)[0]
    fig = Figure()
    FigureCanvasAgg(fig)
    fig.add_subplot(311, projection=ccrs.Mollweide())
    fig.add_subplot(312, projection=ccrs.Orthographic())
    fig.add_subplot(313, projection=ccrs.PlateCarree())
    for ax, col in zip(fig.axes, ['g', 'r', 'b']):
        kde.plot(ax, col)
        kde.plot_decra_samples(ax)


def test_kde_normalised():
    numpy.random.seed(seed=0)
    kde = random_kde(100)[0]
    N = spherical_integrate(kde, log=True)
    assert_allclose(N, 1)


def test_kde_correct():
    numpy.random.seed(seed=0)
    kde, phi0, theta0, sigma0 = random_kde(100)
    kde1 = spherical_kde.SphericalKDE(numpy.mod(kde.phi+numpy.pi, 2*numpy.pi),
                                      numpy.pi-kde.theta)

    def logq(phi, theta):
        return VonMisesFisher_distribution(phi, theta, phi0, theta0, sigma0)

    # Test that the kullback liebler divergence is sufficiently small
    KL = spherical_kullback_liebler(kde, logq)
    assert_allclose(KL, 0, atol=0.1)

    # Null test to see that a completely different KDE is not the same
    KL1 = spherical_kullback_liebler(kde1, logq)
    assert KL1 > 0.1
