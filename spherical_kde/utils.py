import numpy

def cartesian_from_spherical(phi, theta):
    x = numpy.sin(theta) * numpy.cos(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(theta)
    return numpy.array([x, y, z])


def decra_to_polar(ra, dec):
    phi = numpy.mod(ra/180.*numpy.pi,2*numpy.pi)
    theta = numpy.pi/2-dec/180.*numpy.pi
    return phi, theta


def polar_to_decra(phi, theta):
    ra = phi * (phi<numpy.pi) +  (phi-2*numpy.pi)*(phi>numpy.pi)
    dec = numpy.pi/2-theta
    return ra/numpy.pi*180, dec/numpy.pi*180


def infinite_sum(f, i0=0, eps=1e-8):
    i = i0
    total = 0.
    while True:
        diff = f(i)
        if diff < eps*total:
            return total
        total += diff
        i += 1

