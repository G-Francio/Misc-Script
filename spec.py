import bisect
import numpy as np
import astropy.units as au
import matplotlib.pyplot as plt

from astropy import constants as aconst
from astropy.stats import sigma_clip
from astropy.table import Table
from copy import deepcopy as dc
# from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

import utilities as utils
import warnings


class generic_spectrum():
    """
    Basic class to mimic Astrocook spectrum structure.
    Attributes are similar, but not really
    """
    @au.quantity_input
    def __init__(self, **kwargs):
        self.x = kwargs.get('wave', None)
        self.y = kwargs.get('flux', None)
        self.dy = kwargs.get('err', None)
        self.name = kwargs.get('name', None)

        self.z_em = kwargs.get('z_em', np.nan)
        self.x_em = (self.z_em + 1) * kwargs.get('line', 121.567 * au.nm)

        self.xmin = kwargs.get('xmin', None)
        self.xmax = kwargs.get('xmax', None)

        if self.x is not None:
            self._xunit = self.x.unit
        else:
            self._xunit = kwargs.get('xunit', None)

        if self.y is not None:
            self._yunit = self.y.unit
        else:
            self._yunit = kwargs.get('yunit', utils.adm)

        self._old_xunit = None
        self._old_yunit = None
        self.set_nm2kms()

    def gen_xmin_xmax(self):
        if self.x is None:
            raise NameError("Generate a x grid!")
        elif self.xmin is not None and self.xmax is not None:
            return 0
        else:
            self.xmin, self.xmax = utils.get_xmin_xmax(self.x)
            return 0

    def set_nm2kms(self):
        if not np.isnan(self.z_em):
            self._nm2kms = [
                (au.nm, au.km / au.s,
                 lambda x: np.log(x / self.x_em.value) *
                 aconst.c.to(au.km / au.s),
                 lambda x: np.exp(x / aconst.c.to(au.km / au.s).value) * self.x_em.value)
            ]
        else:
            self._nm2kms = utils.default_nm2kms

    def convert_x(self, to, equiv=None):
        if equiv is None:
            equiv = self._nm2kms

        self.gen_xmin_xmax()

        self._old_xunit = self.x.unit

        self.x = self.x.to(to, equivalencies=equiv)
        self.xmin = self.xmin.to(to, equivalencies=equiv)
        self.xmax = self.xmax.to(to, equivalencies=equiv)
        self._xunit = to
        return self.x

    def convert_y(self, to):
        self._old_xunit = self.y.unit
        self.y = self.y.to(to)
        self.dy = self.dy.to(to)

    @au.quantity_input
    def region_extract(self, xmin, xmax, xunit=None, in_place=True):
        assert xmax.value > xmin.value
        if xunit is None:
            xunit = self._xunit

        self.convert_x(xunit)
        inds = np.where((self.x > xmin) & (self.x < xmax))

        new_wave = self.x[inds]
        new_flux = self.y[inds]
        new_err = self.dy[inds]

        if in_place:
            self.x = new_wave
            self.y = new_flux
            self.dy = new_err
            self.xmin, self.xmax = utils.get_xmin_xmax(new_wave)
        else:
            return generic_spectrum(wave=new_wave, flux=new_flux, err=new_err, z_em=self.z_em)

    @au.quantity_input
    def rebin(self, dv, xstart=None, xend=None, filling=np.nan, in_place=True, equiv=None):
        if equiv is None:
            equiv = self._nm2kms

        # Wave array HAS to be sorted, otherwise we have issues (not only on the rebinning...)
        assert all(self.x == np.sort(self.x))
        self.gen_xmin_xmax()
        if dv.unit != utils.kms:
            warnings.warn("dv units not km/s, converting...")
            dv.to(utils.kms, equivalencies=self._nm2kms)

        if self.x.unit != utils.kms:
            warnings.warn("Spec x units not km/s, converting...")
            self.convert_x(utils.kms)

        # Always rebin in velocity space - we check this just above
        if not (xstart is None or xend is None):
            xstart = xstart.to(utils.kms, equivalencies=equiv)
            xend = xend.to(utils.kms, equivalencies=equiv)
        else:
            # Create x, xmin, and xmax for rebinning
            if xstart is None:
                xstart = np.nanmin(self.x)
            if xend is None:
                xend = np.nanmax(self.x)

        x_r = np.arange(xstart.value, xend.value, dv.value) * self._xunit
        xmin_r, xmax_r = utils.get_xmin_xmax(x_r)

        # Compute y and dy combining contributions
        im = 0
        iM = 1

        y_r = np.array([]) * self.y.unit
        dy_r = np.array([]) * self.y.unit

        printval = self.name if self.name is not None else "Spectrum"
        for _, (m, M) in utils.enum_tqdm(zip(xmin_r.value, xmax_r.value), len(x_r), printval + ": Rebinning"):
            im = bisect.bisect_left(np.array(self.xmax.value), m)
            iM = bisect.bisect_right(np.array(self.xmin.value), M)

            ysel = self.y[im:iM]
            dysel = self.dy[im:iM]
            frac = (np.minimum(
                M, self.xmax.value[im:iM]) - np.maximum(m, self.xmin.value[im:iM]))/dv.value

            nw = np.where(~np.isnan(ysel))
            ysel = ysel[nw]
            dysel = dysel[nw]
            frac = frac[nw]

            w = np.where(frac > 0)

            if len(frac[w]) > 0:
                weights = (frac[w] / dysel[w]**2).value
                # and False:
                if np.any(np.isnan(dysel)) or np.any(dysel == 0.0) or np.sum(weights) == 0.0:
                    y_r = np.append(y_r, np.average(ysel[w], weights=frac[w]))
                else:
                    y_r = np.append(y_r, np.average(ysel[w], weights=weights))
                dy_r = np.append(dy_r, np.sqrt(
                    np.nansum(weights**2 * dysel[w].value**2))/np.nansum(weights) * self.y.unit)
            else:
                y_r = np.append(y_r, filling)
                dy_r = np.append(dy_r, filling)

        if in_place:
            self.x = x_r
            self.xmin, self.xmax = xmin_r, xmax_r
            self.convert_x(self._old_xunit)
            self.y = y_r
            self.dy = dy_r
            return 0
        else:
            return generic_spectrum(wave=x_r, flux=y_r, err=dy_r)

    @au.quantity_input
    def sigma_clip(self, window_length=250*utils.kms, in_place=True, **kwargs):
        # Get indexes for each sublist
        def get_indexes(arr, window_in_pix):
            for i in range(len(arr) - window_in_pix + 1):
                yield i, i + window_in_pix

        # Work in velocity space, it's easier
        self.convert_x(utils.kms, equiv=self._nm2kms)
        binsize = np.mean(self.xmax - self.xmin)
        window_in_pix = int(np.round(window_length/binsize).value)

        # Mask for the clipping
        mask = np.full(self.y.size, False)
        # Start from beginning, iterate over till the end
        for start, end in get_indexes(self.y, window_in_pix):
            mask[start:end] = sigma_clip(self.y[start:end], **kwargs).mask

        if in_place:
            self.y[np.where(mask)] = np.nan
            self.dy[np.where(mask)] = np.nan
        else:
            outspec = dc(self)
            outspec.y[np.where(mask)] = np.nan
            outspec.dy[np.where(mask)] = np.nan
            return outspec

    # Moved flux_ccf to spec_manipulation

    def fix_issues(self, sigma=3):
        # Fix spectrum by setting things to NaN

        # I assume that everything that has dy less than zero is useless
        inds = np.where(self.dy <= 0)
        self.y[inds] = np.nan
        self.dy[inds] = np.nan

        # Error to large -> clip away
        inds = np.where(self.dy > sigma * np.nanmedian(self.dy))
        self.y[inds] = np.nan
        self.dy[inds] = np.nan

    def plot(self, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12/1.61))
        utils.plot_spec(ax, self, **kwargs)
        return 0

    def save(self, path, format='fits', overwrite=False):
        t = Table([self.x.value, self.y.value, self.dy.value],
                  names=["Wave", "Flux", "F_err"])
        t.write(path, format='fits', overwrite=overwrite)


# generate mock spectra
#  define a child class of generic spec that inherits everything
#  this just sets some extra methods I can use to have an easier life
class mock_spec(generic_spectrum):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.snr = kwargs.get('snr', 100)  # SNR
        # resolution - should be twice the dispersion
        self.R = kwargs.get('R', 1e5)
        # Synth line sigma (assuming it's a gaussian)
        self.sigma = kwargs.get('sigma', 2*utils.kms)
        # Dispersion of the spectrum
        self.dispersion = kwargs.get('dispersion', 1.5*utils.kms)
        self.npix = kwargs.get('npix', 50)  # number of pixels of the mock
        self.amp = kwargs.get('amp', 0.7)  # depth of the line
        self.pos = kwargs.get('pos', 0 * utils.kms)  # position of the line
        # noise added to the spectrum
        self.noise_amp = kwargs.get('noise_amp', 1/kwargs.get('snr', 100))
        # These should stay None unless I have explicitely smoothed
        #  or splined one of the two spectra. The only exception
        #  for now is in the generation of tracker/Liske points
        # There I already have the y_smooth, so we give it directly
        self.y_smooth = None
        self.dy_smooth = None
        self.y_spline = None
        self.dy_spline = None

    # NB: SNR matches, for this case, the noise_amp value.
    # If you use different noise, this might be different - and tbh, I am not sure
    #  this is actually what I want

    def generate(self, vgrid=None):
        if vgrid is None:
            vgrid = np.arange(-self.npix // 2, self.npix // 2,
                              1) * self.dispersion + self.dispersion/2

        cont = np.full(vgrid.shape, 1.) * utils.adm
        noise = np.random.normal(0., self.noise_amp, size=cont.shape)
        pcont = cont + noise

        if not utils.is_iterable(self.pos):
            syn_f = cont + noise - utils.gaussian(vgrid, self.pos, self.sigma) * (
                np.sqrt(2 * np.pi) * self.sigma.value) * self.amp
            # ^ this is just to set the correct depth of the gaussian, otherwise it's too shallow
        else:
            if not utils.is_iterable(self.amp):
                amp = np.ones(len(self.pos)) * self.amp
            else:
                amp = self.amp
            if not utils.is_iterable(self.sigma):
                sigma = np.ones(len(self.pos)) * self.sigma
            else:
                sigma = self.sigma

            for (p, a, s) in zip(self.pos, amp, sigma):
                cont *= (1. - utils.gaussian(vgrid, p, s) *
                         (np.sqrt(2 * np.pi) * s.value) * a)

            syn_f = cont + noise

        self.x = vgrid
        self.y = syn_f
        self.dy = pcont/self.snr  # do I want np.abs(noise)? or is this good?
        # anyway, there surely is a better way to do this...

    def corr_err(self, corrfunc, kind=3):
        if callable(corrfunc):
            self.dy_corr = self.dy * corrfunc(self.x.value)
            if self.dy_smooth is not None:
                self.dy_smooth_corr = self.dy_smooth * corrfunc(self.x.value)
                _, self.dy_spline_corr = utils.interpolate_flux(
                    self.x, self.y, self.dy_corr, kind=kind)
        else:
            self.dy_corr = self.dy * corrfunc
            if self.dy_smooth is not None:
                self.dy_smooth_corr = self.dy_smooth * corrfunc
                _, self.dy_spline_corr = utils.interpolate_flux(
                    self.x, self.y, self.dy_corr, kind=kind)

    def smooth(self, npix):
        if self.x.unit != utils.kms:
            warnings.warn("x units not km/s, converting...")
            self.convert_x(utils.kms)

        if self.y_smooth is not None:
            warnings.warn("Already smoothed, skipping.")

        # smooth the spectrum, based on the pixel size
        #  and the number of pixels you want to use
        # the important part is to reinterpolate the sigma
        #  on a the grid with the pixel size of the spectrum
        #  otherwise I end up interpolating with a window
        #  that is not what I want but is off by some self.dispersion
        # small optimization, hardcoding constants
        t_sqrt_2_ln_2 = 2 * np.sqrt(2 * np.log(2))
        sqrt_2_ln_100 = np.sqrt(2*np.log(100))

        # get the smoothing sigma
        smoothing_sigma = npix * self.dispersion.value / t_sqrt_2_ln_2

        # not really following scipy for the window definition,
        #  but I *really* want this to be centered around zero
        # and this is heavily inspired by Barak, in the previous
        #  implementation there were edge effects due to the gaussian
        #  being cut too short.
        # So, in order:
        # 1 - Produce the convolution kernel
        #  this computes the kernel till it reaches 1/100 of the max
        n = np.ceil(sqrt_2_ln_100 * smoothing_sigma).astype(int)
        x = np.linspace(-n, n, 2 * n+1) * self.dispersion.value
        kernel = norm.pdf(x, 0., smoothing_sigma)
        # 2 - Normalize the kernel
        kernel /= kernel.sum()
        # 3 - Find the number of pixels to keep later one
        n_keep = len(kernel)
        # 4 - Do the actual convolution
        self.y_smooth = np.convolve(
            utils.pad_invert(self.y, kernel), kernel)[n_keep-1:-n_keep+1] * utils.adm
        self.dy_smooth = np.convolve(
            utils.pad_invert(self.dy, kernel), kernel)[n_keep-1:-n_keep+1] * utils.adm

    def spline(self, npix, kind=3):
        self.smooth(npix)
        self.y_spline, self.dy_spline = utils.interpolate_flux(
            self.x, self.y_smooth, self.dy, kind=kind)

    @au.quantity_input
    def region_extract(self, xmin, xmax, xunit=None, in_place=True):
        assert xmax.value > xmin.value
        if xunit is None:
            xunit = self._xunit

        self.convert_x(xunit)
        inds = np.where((self.x > xmin) & (self.x < xmax))

        new_wave = self.x[inds]
        new_flux = self.y[inds]
        new_err = self.dy[inds]

        if in_place:
            self.x = new_wave
            self.y = new_flux
            self.dy = new_err
            self.xmin, self.xmax = utils.get_xmin_xmax(new_wave)
        else:
            return mock_spec(wave=new_wave, flux=new_flux, err=new_err, z_em=self.z_em)
