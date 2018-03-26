import spherical_kde.utils as utils
import numpy
import numpy.testing
from numpy import pi, sqrt

test_polar = numpy.array([[pi/3, pi/3],
                          [pi/6, pi/4],
                          [pi/4, pi/4]])

test_decra = numpy.array([[60., 30.],
                          [30., 45.],
                          [45., 45.]])

test_cartesian = numpy.array([[sqrt(3.)/4, 3./4, 1./2],
                              [sqrt(3./8), 1./sqrt(8.), 1/sqrt(2.)],
                              [1./2, 1./2, 1/sqrt(2.)]])


def test_cartesian_from_polar_scalar():
    for ang, cart0 in zip(test_polar, test_cartesian):
        cart1 = utils.cartesian_from_polar(*ang)
        numpy.testing.assert_allclose(cart0, cart1)


def test_cartesian_from_polar_array():
    cart1 = utils.cartesian_from_polar(*test_polar.T)
    numpy.testing.assert_allclose(test_cartesian.T, cart1)


def test_polar_from_cartesian_scalar():
    for ang0, cart in zip(test_polar, test_cartesian):
        # Test straightforward
        ang1 = utils.polar_from_cartesian(cart)
        numpy.testing.assert_allclose(ang0, ang1)

        # Test normalisation doesn't matter
        cart = 3*numpy.array(cart)
        ang2 = utils.polar_from_cartesian(cart)
        numpy.testing.assert_allclose(ang0, ang2)


def test_polar_from_cartesian_array():
    ang1 = utils.polar_from_cartesian(test_cartesian.T)
    numpy.testing.assert_allclose(test_polar.T, ang1)


def test_decra_from_polar_scalar():
    for ang, decra0 in zip(test_polar, test_decra):
        decra1 = utils.decra_from_polar(*ang)
        numpy.testing.assert_allclose(decra0, decra1)


def test_decra_from_polar_array():
    decra1 = utils.decra_from_polar(*test_polar.T)
    numpy.testing.assert_allclose(test_decra.T, decra1)


def test_polar_from_decra_scalar():
    for ang0, decra in zip(test_polar, test_decra):
        ang1 = utils.polar_from_decra(*decra)
        numpy.testing.assert_allclose(ang0, ang1)


def test_polar_from_decra_array():
    polar1 = utils.polar_from_decra(*test_decra.T)
    numpy.testing.assert_allclose(test_polar.T, polar1)


def test_logsinh():
    numpy.random.seed(seed=0)
    for i in range(100):
        if i > 90:
            x = numpy.random.rand(10)
        else:
            x = numpy.random.rand()
        a = utils.logsinh(x)
        b = numpy.log(numpy.sinh(x))
        numpy.testing.assert_allclose(a, b)
