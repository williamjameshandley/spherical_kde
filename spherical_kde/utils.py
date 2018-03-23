import numpy

def cartesian_from_spherical(phi, theta):
    x = numpy.sin(theta) * numpy.cos(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(theta)
    return numpy.array([x, y, z])


def decra_to_polar(ra, dec):
    phi = numpy.mod(ra,2*numpy.pi)
    theta = numpy.pi/2-dec
    return phi, theta


def polar_to_decra(phi, theta):
    ra = phi * (phi<numpy.pi) +  (phi-2*numpy.pi)*(phi>numpy.pi)
    dec = numpy.pi/2-theta
    return ra, dec


def infinite_sum(f, i0=0, eps=1e-8):
    i = i0
    total = 0.
    while True:
        diff = f(i)
        if diff < eps*total:
            return total
        total += diff
        i += 1

