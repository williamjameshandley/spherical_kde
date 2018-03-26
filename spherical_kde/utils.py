import numpy


def cartesian_from_polar(phi, theta):
    """ Embedded 3D unit vector from spherical polar coordinates.

    Args:
        phi (float or numpy.array)
            azimuthal angle in radians

        theta (float or numpy.array)
            polar angle in radians

    Returns:
        nhat (numpy.array)
            unit vector(s) in direction phi, theta
    """
    x = numpy.sin(theta) * numpy.cos(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(theta)
    return numpy.array([x, y, z])


def polar_from_cartesian(x):
    """ Embedded 3D unit vector from spherical polar coordinates.

    Args:
        phi (float or numpy.array)
            azimuthal angle in radians

        theta (float or numpy.array)
            polar angle in radians

    Returns:
        nhat (numpy.array)
            unit vector(s) in direction phi, theta
    """
    x = numpy.array(x)
    r = (x*x).sum(axis=0)**0.5
    x, y, z = x
    theta = numpy.arccos(z / r)
    phi = numpy.arctan2(y, x)
    return phi, theta


def polar_from_decra(ra, dec):
    """ Convert from spherical polar coordinates to ra and dec.

    Args:
        ra (float or numpy.array)
            Right ascension in degrees

        dec (float or numpy.array)
            Declination in degrees

    Returns:
        phi, theta (float or numpy.array)
            Spherical polar coordinates in radians
    """
    phi = numpy.mod(ra/180.*numpy.pi, 2*numpy.pi)
    theta = numpy.pi/2-dec/180.*numpy.pi
    return phi, theta


def decra_from_polar(phi, theta):
    """ Convert from ra and dec to spherical polar coordinates.

    Args:
        phi (float or numpy.array)
            azimuthal angle in radians

        theta (float or numpy.array)
            polar angle in radians

    Returns:
        phi, theta (float or numpy.array)
            Right ascension and declination in degrees
    """
    ra = phi * (phi < numpy.pi) + (phi-2*numpy.pi)*(phi > numpy.pi)
    dec = numpy.pi/2-theta
    return ra/numpy.pi*180, dec/numpy.pi*180


def logsinh(x):
    """ Compute log(sinh(x)), stably for large x.

    Args:
        x (float or numpy.array)
            argument to evaluate at, must be positive
    Returns:
        log(sinh(x)) (float or numpy.array)
    """
    if numpy.any(x < 0):
        raise ValueError("logsinh only valid for positive arguments")
    return x + numpy.log(1-numpy.exp(-2*x)) - numpy.log(2)
