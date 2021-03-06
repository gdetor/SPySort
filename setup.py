# -*- coding: utf-8 -*-
from setuptools import setup

from spysort.version import version

long_description = open("README.md").read()

install_requires = ['numpy>=1.3.0', 'pandas>=0.12.0', 'scipy>=0.9.0',
                    'matplotlib>=1.1.0', 'sqlalchemy>=0.7',
                    'scikit-learn>=0.11', ]

setup(name="spysort",
      version=version,
      packages=['spysort', 'spysort.ReadData',
                'spysort.Events', 'examples', 'doc'],
      include_package_data=True,
      install_requires=install_requires,
      requires=[],
      author="C. Pouzat and G.Is. Detorakis",
      author_email="gdetor@gmail.com",
      maintainer="Georgios Is. Detorakis",
      maintainer_email="gdetor@gmail.com",
      long_description=long_description,
      url='https://github.com/gdetor/SPySort/',
      license="BSD",
      description="SPySort: Spike Sorting toolbox",
      classifiers=['Development Status :: 1 - Alpha',
                   'Intended Audience :: Neuroscience/Research',
                   'License :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2',
                   'Topic :: Scientific/Engineering :: Neuroscience'
                   ], )
