from spherical_kde.distributions import VonMisesFisher_distribution as VMF
from spherical_kde.utils import cartesian_from_polar, polar_from_cartesian
import numpy
from numpy.testing import assert_allclose

from scipy.integrate import dblquad


def random_VonMisesFisher_distribution():
    phi0 = numpy.random.rand()*numpy.pi*2
    theta0 = numpy.random.rand()*numpy.pi
    sigma = numpy.random.rand()

    def f(phi, theta):
        return numpy.exp(VMF(phi, theta, phi0, theta0, sigma))
    return f, phi0, theta0


def spherical_integrate(f):
    ans, _ = dblquad(lambda phi, theta: f(phi, theta)*numpy.sin(theta),
                     0, numpy.pi, lambda x: 0, lambda x: 2*numpy.pi)
    return ans


def test_VonMisesFisher_distribution_normalisation():
    numpy.random.seed(seed=0)
    for _ in range(3):
        f, phi0, theta0 = random_VonMisesFisher_distribution()
        N = spherical_integrate(f)
        assert_allclose(N, 1)


def test_VonMisesFisher_distribution_mean():
    numpy.random.seed(seed=0)
    for _ in range(3):
        f, phi0, theta0 = random_VonMisesFisher_distribution()
        x = []
        for i in range(3):
            def g(phi, theta):
                return f(phi, theta) * cartesian_from_polar(phi, theta)[i]
            x.append(spherical_integrate(g))
        phi, theta = polar_from_cartesian(x)
        assert_allclose([phi0, theta0], [phi, theta])
