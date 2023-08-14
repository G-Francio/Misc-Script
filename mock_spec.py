import numpy as np
import astropy.units as au
import matplotlib.pyplot as plt

from astropy import constants as aconst
from lmfit import minimize, Parameters
from astropy.io import fits

import utilities as utils
import barak as bk
import spec as gspec

import warnings

# generate mock spectra
#  define a child class of generic spec

class mock_spec(gspec.generic_spectrum):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.snr = kwargs.get('snr', 100) # SNR
        self.R   = kwargs.get('R', 7e4) # resolution
        self.sigma = kwargs.get('sigma', 2*utils.kms) # Synth line sigma (assuming it's a gaussian)
        self.dispersion = kwargs.get('dispersion', 1.5*utils.kms) # Dispersion of the spectrum
        self.npix = kwargs.get('npix', 50) # number of pixels of the mock
        self.amp = kwargs.get('amp', 0.7) # depth of the line
        self.pos = kwargs.get('pos', 0 * utils.kms) # position of the line
        self.noise_amp = kwargs.get('noise_amp', 1/kwargs.get('snr', 100)) # noise added to the spectrum
        # These should stay None unless I have explicitely splined one of the two spectra.
        self.y_smooth  = None
        self.dy_smooth = None
        self.y_spline  = None
        self.dy_spline = None
            
        
    def generate(self):
        vgrid = np.arange(-self.npix // 2, self.npix //2, 1) * self.dispersion
        cont  = np.full(vgrid.shape, 1.) * utils.adm
        pcont = cont + np.random.normal(0., self.noise_amp, size = cont.shape)
        
        self.x = vgrid
        self.y = syn_f
        self.dy = pcont/self.snr # there's probably a better way to do this...
        
        
    def smooth(self, npix):
        if self.x.unit != utils.kms:
            warnings.warn("x units not km/s, converting...")
            self.convert_x(utils.kms)

        self.y_smooth  = bk.convolve_psf(self.y, npix) * utils.adm
        self.dy_smooth = bk.convolve_psf(self.dy, npix) * utils.adm
        
    
    def spline(self, npix, kind = 3):
        self.smooth(npix)
        self.y_spline, self.dy_spline = utils.interpolate_flux(self.x, self.y_smooth, self.dy_smooth, kind = kind)

