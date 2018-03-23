import matplotlib
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from spherical_kde.distributions import VMF
from spherical_kde.utils import polar_to_decra, decra_to_polar

%matplotlib
from scipy.stats import ortho_group
import numpy

nsamples = 100
theta_samples = 1+numpy.random.normal(size=nsamples)*0.3
phi_samples = 1+numpy.random.normal(size=nsamples)*0.3 
def kde(phi, theta, sigma):
    return logsumexp(VMF(phi, theta, phi_samples, theta_samples, sigma),axis=-1) - numpy.log(len(theta_samples))

fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')

dec_samples, ra_samples = polar_to_decra(theta_samples, phi_samples)
ax.plot(ra_samples, dec_samples, 'k.')

ra = numpy.linspace(-numpy.pi, numpy.pi, 100)
dec = numpy.linspace(-numpy.pi/2, numpy.pi/2, 100)
X, Y = numpy.meshgrid(ra, dec)

phi, theta = decra_to_polar(dec, ra)
P = numpy.exp(kde(phi, theta, 0.2)) 

ax.contour(X, Y, P, 3, colors='k')

