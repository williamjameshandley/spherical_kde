import spherical_kde.utils as utils
import numpy
import numpy.testing
from numpy import pi, sqrt

def test_cartesian_from_spherical_scalar():
    """ Test transformation from cartesian to spherical """
    a = utils.cartesian_from_spherical(pi/3, pi/3)
    b = [sqrt(3.)/4,3./4,1./2] 
    numpy.testing.assert_allclose(a,b)

    a = utils.cartesian_from_spherical(pi/6, pi/4)
    b = [sqrt(3./8),1./sqrt(8.),1/sqrt(2.)] 
    numpy.testing.assert_allclose(a,b)
