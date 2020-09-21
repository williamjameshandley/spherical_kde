#!/usr/bin/env python3

from distutils.core import setup
try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError):
    long_description = ''

setup(name='spherical_kde',
      version='0.1.2',
      author='Will Handley',
      author_email='wh260@cam.ac.uk',
      url='https://github.com/williamjameshandley/spherical_kde',
      download_url = 'https://github.com/williamjameshandley/spherical_kde/archive/0.0.6.tar.gz',
      packages=['spherical_kde', 'spherical_kde.tests'],
      install_requires=['cartopy', 'pytest', 'numpy', 'scipy', 'matplotlib', 'pypandoc', 'numpydoc'],
      license='MIT',
      classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Natural Language :: English',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3.6',
      'Topic :: Scientific/Engineering :: Astronomy',
      'Topic :: Scientific/Engineering :: Physics',
      'Topic :: Scientific/Engineering :: Visualization',
      'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      description='Kernel density estimation on a sphere',
      long_description=long_description
      )
