import numpy
from numpy.testing import assert_allclose
import spherical_kde.distributions as dxns
from spherical_kde.utils import (cartesian_from_polar, polar_from_cartesian,
                                 spherical_integrate)


def random_phi_theta_sigma():
    phi = numpy.random.rand()*numpy.pi*2
    theta = numpy.random.rand()*numpy.pi
    sigma = numpy.random.rand()
    return phi, theta, sigma


def random_VonMisesFisher_distribution():
    phi0, theta0, sigma0 = random_phi_theta_sigma()

    def f(phi, theta):
        return numpy.exp(dxns.VonMisesFisher_distribution(phi, theta,
                                                          phi0, theta0,
                                                          sigma0))
    return f, phi0, theta0, sigma0


def test_VonMisesFisher_distribution_normalisation():
    numpy.random.seed(seed=0)
    for _ in range(3):
        f, phi0, theta0, sigma0 = random_VonMisesFisher_distribution()
        N = spherical_integrate(f)
        assert_allclose(N, 1)


def test_VonMisesFisher_distribution_mean():
    numpy.random.seed(seed=0)
    for _ in range(3):
        f, phi0, theta0, sigma0 = random_VonMisesFisher_distribution()
        x = []
        for i in range(3):
            def g(phi, theta):
                return f(phi, theta) * cartesian_from_polar(phi, theta)[i]
            x.append(spherical_integrate(g))
        phi, theta = polar_from_cartesian(x)
        assert_allclose([phi0, theta0], [phi, theta])


def test_VonMisesFisher_mean():
    numpy.random.seed(seed=0)
    for _ in range(3):
        phi0, theta0, sigma0 = random_phi_theta_sigma()
        N = 10000
        phi, theta = dxns.VonMisesFisher_sample(phi0, theta0, sigma0, N)
        phi, theta = dxns.VonMises_mean(phi, theta)
        assert_allclose((phi0, theta0), (phi, theta), 1e-2)


def test_VonMisesFisher_standarddeviation():
    numpy.random.seed(seed=0)
    for _ in range(3):
        phi0, theta0, sigma0 = random_phi_theta_sigma()
        N = 10000
        phi, theta = dxns.VonMisesFisher_sample(phi0, theta0, sigma0, N)
        sigma = dxns.VonMises_standarddeviation(phi, theta)
        assert_allclose(sigma0, sigma, 1e-2)
