#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages


import os
from setuptools import setup
from setuptools.command.install import install
import subprocess

__minimum_numpy_version__ = '1.9.0'

setup_requires = ['numpy>='+__minimum_numpy_version__, 'astropy','mippy']

setup(name='rathings',
      version='0.0.1',
      description='Radio Astronomy Things',
      author=['Josh Albert'],
      author_email=['albert@strw.leidenuniv.nl'],
##      url='https://www.python.org/sigs/distutils-sig/',
    setup_requires=setup_requires,  
    tests_require=[
        'pytest>=2.8',
    ],
    package_dir = {'':'src'},
      packages=find_packages('src')
     )

