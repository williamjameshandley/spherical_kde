import spherical_kde.utils as utils
import numpy
import numpy.testing
from numpy import pi, sqrt

test_spherical = numpy.array([[pi/3, pi/3],
                              [pi/6, pi/4],
                              [pi/4, pi/4]])

test_cartesian = numpy.array([[sqrt(3.)/4, 3./4, 1./2],
                              [sqrt(3./8), 1./sqrt(8.), 1/sqrt(2.)],
                              [1./2, 1./2, 1/sqrt(2.)]])


def test_cartesian_from_spherical_scalar():
    """ Test transformation from cartesian to spherical """
    for ang, cart0 in zip(test_spherical, test_cartesian):
        cart1 = utils.cartesian_from_spherical(*ang)
        numpy.testing.assert_allclose(cart0, cart1)


def test_cartesian_from_spherical_array():
    """ Test transformation from cartesian to spherical """
    cart1 = utils.cartesian_from_spherical(*test_spherical.T)
    numpy.testing.assert_allclose(test_cartesian.T, cart1)


def test_spherical_from_cartesian_scalar():
    """ Test transformation from spherical to cartesian """
    for ang0, cart in zip(test_spherical, test_cartesian):
        # Test straightforward
        ang1 = utils.spherical_from_cartesian(cart)
        numpy.testing.assert_allclose(ang0, ang1)

        # Test normalisation doesn't matter
        cart = 3*numpy.array(cart)
        ang2 = utils.spherical_from_cartesian(cart)
        numpy.testing.assert_allclose(ang0, ang2)


def test_spherical_from_cartesian_array():
    """ Test transformation from cartesian to spherical """
    ang1 = utils.spherical_from_cartesian(test_cartesian.T)
    numpy.testing.assert_allclose(test_spherical.T, ang1)
