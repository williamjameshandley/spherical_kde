import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import logsumexp
from spherical_kde.distributions import VMF
from spherical_kde.utils import cartesian_from_spherical

%matplotlib
from scipy.stats import ortho_group
import numpy



theta = numpy.linspace(0, numpy.pi, 100)
phi = numpy.linspace(0, 2*numpy.pi, 100)
theta, phi = numpy.meshgrid(theta, phi)


#g = ortho_group.rvs(dim=3)
#k = KentDistribution()
#fcolors = numpy.exp(k(theta, phi, 100, 0.00, *g))




nsamples = 1000
theta_samples = numpy.random.uniform(low=numpy.pi/4,high=3*numpy.pi/4,size=nsamples)
phi_samples = numpy.random.uniform(low=0,high=numpy.pi/2,size=nsamples)
def kde(theta, phi, sigma):
    return logsumexp(VMF(theta, phi, theta_samples, phi_samples, sigma),axis=-1) - numpy.log(len(theta_samples))

#ax.plot(x,y,z,'k.')

fcolors = numpy.exp(kde(theta, phi, 0.05))
fmax, fmin = fcolors.max(), fcolors.min()
fcolors = (fcolors - fmin)/(fmax - fmin)

x, y, z = cartesian_from_spherical(theta, phi)

fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=matplotlib.cm.seismic(fcolors))
