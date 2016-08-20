
# coding: utf-8

# In[ ]:

import numpy as np

def fft( data ):
    return numpy.fft.fftshift( numpy.fft.fftn( numpy.fft.ifftshift( data ) ) )

def ifft( data ):
    return numpy.fft.fftshift( numpy.fft.ifftn( numpy.fft.ifftshift( data ) ) )

'''Propagate Gaussians from sky model (a sum of point or gaussians) to the upper layers of the ionosphere,
and then onward through to the array.
We can propagate the intensity and the auto-correlation. Both have nice features in fourier propagation.'''


