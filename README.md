Spherical Kernel Density Estimation
===================================

These packages allow you to do rudimentary kernel density estimation on a
sphere. Extreme alpha development status.

The fundamental principle is to replace the traditional Gaussian function used
in 
[kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation)
with the
[Von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution).
We then plot on a Mollweide projection, as this is equal area, and therefore
has the least visual disturbance regarding PDF estimation.

Example Usage
-------------

```python
import numpy
from spherical_kde.kde import SphericalKDE
from spherical_kde.utils import polar_to_decra, decra_to_polar

# Generate some samples centered on (1,1)
nsamples = 100
theta_samples = numpy.random.normal(loc=1,scale=0.3,size=nsamples)
phi_samples = numpy.random.normal(loc=1,scale=0.3,size=nsamples)
phi_samples = numpy.mod(phi_samples,numpy.pi*2)

kde_0 = SphericalKDE(phi_samples, theta_samples)

# Plot green contours
fig, ax = kde_0.plot('g')

# Plot the actual samples on top
ra_samples, dec_samples = polar_to_decra(kde_0.phi, kde_0.theta)
ax.plot(ra_samples, dec_samples, 'k.')

# Generate some more samples and plot on the same axis in red
phi_samples = numpy.random.normal(loc=-1,scale=0.3,size=nsamples)
phi_samples = numpy.mod(phi_samples,numpy.pi*2)
kde_1 = SphericalKDE(phi_samples, theta_samples)
kde_1.plot('r', ax=ax)


# Generate different samples in blue
theta_samples = numpy.random.normal(loc=2,scale=0.1,size=nsamples)
phi_samples = numpy.random.rand(nsamples)*numpy.pi*2
kde_2 = SphericalKDE(phi_samples, theta_samples)
kde_2.plot('b', ax=ax)

# Save to plot
fig.savefig('plot.png')

```
![](https://raw.github.com/williamjameshandley/spherical_kde/master/plot.png)

To do
-----
* [ ] Bandwidth estimation
* [ ] Other projections
* [ ] Change centre of Mollweide projection
