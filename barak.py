# Barak functions, converted to Py3 and adapted to support astropy quantities

import numpy as np
import astropy.units as au
from astropy import constants as aconst

kms = au.km / au.s
adm = au.dimensionless_unscaled
c_kms = aconst.c.to(kms).value


def convolve_constant_dv(wa, fl, wa_dv=None, npix=4., vfwhm=None):
    """ Convolve a wavelength array with a gaussian of constant
    velocity width.

    If `vfwhm` is specified an intermediate wavelength array with
    constant velocity pixel width is calculated. Otherwise, both
    `wa_dv` and `npix` must be given -- this is faster because no
    intermediate array needs to be calculated.

    Parameters
    ----------
    fl, wa : arrays of floats, length N
      The array to be convolved and its wavelengths.
    vfwhm : float, optional
      Full width at half maximum in velocity space (km/s) of the
      gaussian kernel with which to convolve `fl`.
    npix : float, default 4
      Number of pixels corresponding to `vfwhm` in `wa_dv` if given,
      otherwise `wa` is interpolated to an array with velocity pixel
      width = vfwhm / npix.
    wa_dv : array of floats, default `None`
      Wavelength array with a constant velocity width (this can be
      generated with make_constant_dv_wa_scale()).

    Returns
    -------
    fl_out : array of length N
      fl convolved with the gaussian kernel with the specified FWHM.

    """
    # interpolate to the log-linear scale, convolve, then
    # interpolate back again.
    # convolve with the gaussian
    if vfwhm is not None:
        wa_dv = make_constant_dv_wa_scale(wa[0], wa[-1], float(vfwhm)/npix)
    fl_dv = np.interp(wa_dv, wa, fl)
    fl_dv_smoothed = convolve_psf(fl_dv, npix)
    fl_out = np.interp(wa, wa_dv, fl_dv_smoothed)
    return fl_out


def convolve_psf(a, fwhm, edge='invert', replace_nan=True, debug=False):
    """ Convolve an array with a gaussian window.

    Given an array of values `a` and a gaussian full width at half
    maximum `fwhm` in pixel units, returns the convolution of the
    array with the normalised gaussian.

    Parameters
    ----------
    a : array, shape(N,)
      Array to convolve
    fwhm : float
      Gaussian full width at half maximum in pixels. This should be > 2
      to sample the gaussian PSF properly.

    Returns
    -------
    convolved_a : array, shape (N,)

    Notes
    -----
    The Gaussian kernel is calculated for as many pixels required
    until it drops to 1% of its peak value. The data will be spoiled
    at distances `n`/2 (rounding down) from the edges, where `n` is
    the width of the Gaussian in pixels.
    """
    const2 = 2.354820046             # 2*sqrt(2*ln(2))
    const100 = 3.034854259             # sqrt(2*ln(100))
    sigma = fwhm / const2
    # gaussian drops to 1/100 of maximum value at x =
    # sqrt(2*ln(100))*sigma, so number of pixels to include from
    # centre of gaussian is:
    n = np.ceil(const100 * sigma).astype(int)
    if replace_nan:
        a = nan2num(a, replace='interp')
    if debug:
        print("First and last %s pixels of output array will be invalid" % n)
    x = np.linspace(-n, n, 2*n + 1)        # total no. of pixels = 2n+1
    gauss = np.exp(-0.5 * (x / sigma) ** 2)

    return convolve_window(a, gauss, edge=edge)


def nan2num(a, replace=0):
    """ Replace `nan` or `inf` entries with the `replace` keyword
    value.

    If replace is "mean", use the mean of the array to replace
    values. If it's "interp", intepolate from the nearest values.
    """
    a = np.atleast_1d(a)
    b = np.array(a, copy=True)
    bad = np.isnan(b) | np.isinf(b)
    if replace == 'mean' and (~bad).sum() > 0:
        replace = b[~bad].mean().astype(b.dtype)
    elif replace == 'interp':
        x = np.arange(len(a))
        replace = np.interp(x[bad], x[~bad], b[~bad]).astype(b.dtype)

    b[bad] = replace
    if len(b) == 1:
        return b[0]
    return b


def make_constant_dv_wa_scale(wmin, wmax, dv):
    """ Make a constant velocity width scale given a start and end
    wavelength, and velocity pixel width.
    """
    dlogw = np.log10(1 + dv/c_kms)
    # find the number of points needed.
    npts = np.ceil(np.log10(wmax / wmin) / dlogw)
    wa = wmin * 10**(np.arange(npts)*dlogw)
    return wa


def convolve_window(a, window, edge='invert'):
    """ Convolve an array with an arbitrary window.

    Parameters
    ----------
    a : array, shape (N,)
    window : array, shape (M,)
      The window array should have an odd number of elements.
    edge : {'invert', 'reflect', 'extend'} or int  (default 'invert')
      How to mitigate edge effects. If 'invert', the edges of `a` are
      extended by inversion, similarly for reflection. 'extend' means
      the intial and final points are replicated to extend the
      array. An integer value means take the median of that many
      points at each end and extend by replicating the median value.

    Returns
    -------
    convolved_a : array, shape (N,)

    Notes
    -----
    The window is normalised before convolution.
    """
    npts = len(window)
    if not npts % 2:
        raise ValueError('`window` must have an odd number of elements!')

    n = npts // 2

    # normalise the window
    window /= window.sum()

    # Add edges to either end of the array to reduce edge effects in
    # the convolution.
    if len(a) < 2*n:
        raise ValueError(
            'Window is too big for the array! %i %i' % (len(a), n))
    if edge == 'invert':
        temp1 = 2*a[0] - a[n:0:-1], a, 2*a[-1] - a[-2:-n-2:-1]
    elif edge == 'reflect':
        temp1 = a[n:0:-1], a, a[-2:-n-2:-1]
    elif edge == 'extend':
        temp1 = a[0] * np.ones(n), a, a[-1] * np.ones(n)
    else:
        try:
            abs(int(edge))
        except TypeError:
            raise ValueError('Unknown value for edge keyword: %s' % edge)
        med1 = np.median(a[:edge])
        med2 = np.median(a[-edge:])
        temp1 = med1 * np.ones(n), a, med2 * np.ones(n)

    temp2 = np.convolve(np.concatenate(temp1), window, mode='same')

    return temp2[n:-n]
