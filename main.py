from spherical_kde.kde import SphericalKDE
from spherical_kde.utils import polar_to_decra, decra_to_polar

%matplotlib
import numpy

nsamples = 10000
theta_samples = 1 + numpy.random.normal(size=nsamples)*0.3
phi_samples = numpy.mod(1 + numpy.random.normal(size=nsamples)*0.3,numpy.pi*2)
kde = SphericalKDE(phi_samples, theta_samples)

fig, ax = kde.plot('g')

theta_samples = 1 + numpy.random.normal(size=nsamples)*0.3
phi_samples = numpy.mod(-1 + numpy.random.normal(size=nsamples)*0.3,numpy.pi*2)
kde = SphericalKDE(phi_samples, theta_samples)
kde.plot('r', ax=ax)

theta_samples = 2 + numpy.random.normal(size=nsamples)*0.1
phi_samples = numpy.mod( + numpy.random.normal(size=nsamples)*10,numpy.pi*2)
kde = SphericalKDE(phi_samples, theta_samples)
kde.plot('b', ax=ax)

#ra_samples, dec_samples = polar_to_decra(kde.phi, kde.theta)
#ax.plot(ra_samples, dec_samples, 'k.')
