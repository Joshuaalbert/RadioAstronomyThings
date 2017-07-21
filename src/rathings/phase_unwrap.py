#!/bin/env python

"""
This unwraps phase
author: Joshua Albert
albert@strw.leidenuniv.nl
"""

import numpy as np

def phase_unwrapp1d(theta,axis=None):
    '''The difference between two timesteps is unaliased by assumption so theta_i+1 - theta_i < pi.
    So Wrap(theta_i+1 - theta_i) is the real gradient and we can integrate them.
    if axis is not None then perform the unwrap down an axis.'''
    def wrap(phi):
        res = 1j*phi
        np.exp(res,out=res)
        return np.angle(res)
    if axis is not None:
        theta_ = np.rollaxis(theta,axis)
        diff = theta[1:,...] - theta[:-1,...]
        grad = wrap(diff)
        unwrapped_theta = np.rollaxis(np.zeros(theta.shape,dtype=np.double),axis)
        np.cumsum(grad,out=unwrapped_theta[1:,...],axis=0)
        unwrapped_theta += theta[0,...]
        unwrapped_theta = np.rollaxis(unwrapped_theta,0,start=axis)
    else:
        diff = theta[1:] - theta[:-1]
        grad = wrap(diff)
        unwrapped_theta = np.zeros(len(theta),dtype=np.double)
        np.cumsum(grad,out=unwrapped_theta[1:])
        unwrapped_theta += theta[0]
    return unwrapped_theta
