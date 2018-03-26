import spherical_kde.utils as utils
import pytest
import numpy
from numpy.testing import assert_allclose

test_polar = numpy.array([[numpy.pi/3, numpy.pi/3],
                          [numpy.pi/6, numpy.pi/4],
                          [numpy.pi/4, numpy.pi/4]])

test_decra = numpy.array([[60., 30.],
                          [30., 45.],
                          [45., 45.]])

test_cartesian = numpy.array([[3.**0.5/4, 3./4, 1./2],
                              [(3./8)**0.5, (8.)**-0.5, (2.)**-0.5],
                              [1./2, 1./2, (2.)**-0.5]])


def test_cartesian_from_polar_scalar():
    for ang, cart0 in zip(test_polar, test_cartesian):
        cart1 = utils.cartesian_from_polar(*ang)
        assert_allclose(cart0, cart1)


def test_cartesian_from_polar_array():
    cart1 = utils.cartesian_from_polar(*test_polar.T)
    assert_allclose(test_cartesian.T, cart1)


def test_polar_from_cartesian_scalar():
    for ang0, cart in zip(test_polar, test_cartesian):
        # Test straightforward
        ang1 = utils.polar_from_cartesian(cart)
        assert_allclose(ang0, ang1)

        # Test normalisation doesn't matter
        cart = 3*numpy.array(cart)
        ang2 = utils.polar_from_cartesian(cart)
        assert_allclose(ang0, ang2)


def test_polar_from_cartesian_array():
    ang1 = utils.polar_from_cartesian(test_cartesian.T)
    assert_allclose(test_polar.T, ang1)


def test_decra_from_polar_scalar():
    for ang, decra0 in zip(test_polar, test_decra):
        decra1 = utils.decra_from_polar(*ang)
        assert_allclose(decra0, decra1)


def test_decra_from_polar_array():
    decra1 = utils.decra_from_polar(*test_polar.T)
    assert_allclose(test_decra.T, decra1)


def test_polar_from_decra_scalar():
    for ang0, decra in zip(test_polar, test_decra):
        ang1 = utils.polar_from_decra(*decra)
        assert_allclose(ang0, ang1)


def test_polar_from_decra_array():
    polar1 = utils.polar_from_decra(*test_decra.T)
    assert_allclose(test_polar.T, polar1)


def test_logsinh():
    numpy.random.seed(seed=0)
    for i in range(100):
        if i > 90:
            x = numpy.random.rand(10)
        else:
            x = numpy.random.rand()
        a = utils.logsinh(x)
        b = numpy.log(numpy.sinh(x))
        assert_allclose(a, b)


def test_logsinh_positive_arg():
    with pytest.raises(ValueError):
        utils.logsinh(-1)
    with pytest.raises(ValueError):
        utils.logsinh(numpy.array([1, -1]))


def test_rotation_matrix():
    numpy.random.seed(seed=0)

    theta = numpy.random.rand(10, 2)*numpy.pi
    phi = numpy.random.rand(10, 2)*2*numpy.pi
    for (p1, p2), (t1, t2) in zip(phi, theta):
        n1 = utils.cartesian_from_polar(p1, t1)
        n2 = utils.cartesian_from_polar(p2, t2)
        M = utils.rotation_matrix(n1, n2)
        assert_allclose(M.dot(n1), n2)
        assert_allclose(M.T.dot(n2), n1)


def test_spherical_integrate():
    ans = utils.spherical_integrate(lambda theta, phi: 1)
    assert_allclose(ans, 4*numpy.pi)

    ans = utils.spherical_integrate(lambda theta, phi: theta*phi)
    assert_allclose(ans, 2*numpy.pi**3)

    ans = utils.spherical_integrate(lambda theta, phi: theta*numpy.cos(phi))
    assert_allclose(ans, 0, atol=1e-7)


def test_spherical_kullback_liebler():
    def logp(phi, theta):
        return numpy.log(numpy.sin(theta)/numpy.pi**2)

    def logq(phi, theta):
        return numpy.log(1/numpy.pi/4)

    assert_allclose(utils.spherical_integrate(logp, log=True), 1)
    assert_allclose(utils.spherical_integrate(logq, log=True), 1)

    KL = utils.spherical_kullback_liebler(logp, logq)
    assert_allclose(KL, 1./2 - numpy.log(numpy.pi/2))
