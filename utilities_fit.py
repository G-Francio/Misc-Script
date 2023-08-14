"""
Module for fitting routines
"""
import numpy as np

import utilities as utils
import spec_manipulation as spec_man

from tqdm import tqdm
from copy import deepcopy as dc

from astropy import units as au
from lmfit import minimize, Parameters


# x should be log(n_smooth_pix) -> NOT in units of the FWHM, just the number of pixels!!!
def chi2_corr_func(x, params=None):
    if params is None:
        return np.nan
    return params['d'] + np.exp(params['a'] * np.log(x)**2 + params['b'] * np.log(x) + params['c'])


def fitting_function(params, wa, fl=None, fl_spl=None, efl=None, efl_spl=None, n_smooth_pix=None, corrfunc=None):
    """
    Computes the modified chi square mentioned in the paper.
    Will be the input to the least square minimization process.
    The wavelength grid has to be given in kms and will be used
    for both the un-splined spectrum and the spline itself.

    Note that due to requirement of LMFit this will not accept
    quantities (to avoid headaches as much as possible, shit
    seems already too complicated on its own).
    Also note that there is a dirty fix - till I figure out
    how to use a modified chi square in LMFit (or I write a custom
    Levenberg-Marquardt for fitting): the sqrt in the
    denominator is used to recover the correct chi^2.
    This ~might~ cause numerical issues. They ~should~ be irrelevant,
    but it is something I have to take a look at at some point.

    Note:

    Parameters
    ----------

    Returns
    ----------
    """
    parvals = params.valuesdict()
    ampl = parvals['ampl']
    shift = parvals['shift']
    # we follow the MM implementation for tilt
    tilt = parvals['tilt']
    scale = ampl + wa * tilt

    # Stop if I don't pass to the function everything needed
    if fl is not None and efl is None and efl_spl is None:
        raise NameError(
            'Incomplete input: check if you are missing flux, error or splines!')
    # if I don't give the un-splined flux this will just return the
    #  modified spline
    if fl is None:
        return scale * fl_spl(wa + shift)
    elif fl is not None and n_smooth_pix is None:
        return (fl - scale * fl_spl(wa + shift))/np.sqrt(efl**2 + (scale * efl_spl(wa + shift))**2)
    elif fl is not None and n_smooth_pix is not None:
        corr_1 = corrfunc(n_smooth_pix[0])
        corr_2 = corrfunc(n_smooth_pix[1])
        return (fl - scale * fl_spl(wa + shift))/np.sqrt(corr_1 * efl**2 + corr_2 * (scale * efl_spl(wa + shift))**2)
        # Note: this correction is a bit weird, and I am not super sure if it's actually what
        #  I am supposed to use for the correction of the un-splined spectrum.


@au.quantity_input
def fit_for_shift(in_1, in_2, shift=0.3 * utils.kms, n_smooth_pix_1=1, n_smooth_pix_2=1, corrfunc=None,
                  xi_sigma=1, xi_fwhm=1, correct_for_smooth=True, correct_for_err=True):
    # initialize parameters
    mc_params = Parameters()
    mc_params.add('ampl',  value=1)
    mc_params.add('shift', value=0)
    mc_params.add('tilt',  value=0)

    # check if you need to generate the spectra, or you can just take input
    if utils.is_iterable(in_1) and utils.is_iterable(in_2):
        mc_s1, mc_s2 = spec_man.gen_spec_pair(
            in_1, in_2, shift=shift, n_smooth_pix_1=n_smooth_pix_1, n_smooth_pix_2=n_smooth_pix_2)
    else:
        mc_s1, mc_s2 = in_1, in_2

    # bunch of things to pass to the fitting function, syntax is weird but it is what it is...
    mc_args = [mc_s1.x.value]
    if correct_for_smooth:
        mc_kwargs = {'fl': mc_s1.y_smooth.value, 'fl_spl': mc_s2.y_spline,
                     'efl': mc_s1.dy.value, 'efl_spl': mc_s2.dy_spline,
                     'n_smooth_pix': [n_smooth_pix_1, n_smooth_pix_2], 'corrfunc': corrfunc}
    else:
        mc_kwargs = {'fl': mc_s1.y_smooth.value, 'fl_spl': mc_s2.y_spline,
                     'efl': mc_s1.dy.value, 'efl_spl': mc_s2.dy_spline,
                     'n_smooth_pix': None, 'corrfunc': None}

    # fit and correct the error
    mc_out = minimize(fitting_function, mc_params,
                      args=mc_args, kws=mc_kwargs)
    if mc_out.params['shift'].stderr is None:
        mc_out.params['shift'].stderr = np.nan
    if correct_for_err:
        # TODO: sqrt(chi2) might be needed -> probably if I tell lmfit to not scale the covariance,
        #  otherwise I *think* it is not needed!
        mc_err_corr = mc_out.params['shift'].stderr / (xi_sigma * xi_fwhm)
    else:
        mc_err_corr = mc_out.params['shift'].stderr

    return mc_out.params['shift'].value, mc_err_corr, mc_out.redchi


@au.quantity_input
def fit_many_times(p_sp_1, p_sp_2, shift=0.3 * utils.kms, rep=1000, n_smooth_pix_1=1, n_smooth_pix_2=1,
                   correct_for_smooth=False, correct_for_err=False, corrfunc=None):
    val = []
    err = []
    xsq = []

    fit_params = Parameters()
    fit_params.add('ampl',  value=1.0)
    fit_params.add('shift', value=0.8)
    fit_params.add('tilt',  value=0.0)

    for _ in range(rep):
        val_out, err_out, xsq_out = fit_for_shift(p_sp_1, p_sp_2, shift=shift,
                                                  n_smooth_pix_1=n_smooth_pix_1,
                                                  n_smooth_pix_2=n_smooth_pix_2,
                                                  correct_for_smooth=correct_for_smooth,
                                                  correct_for_err=correct_for_err,
                                                  corrfunc=corrfunc)

        val.append(val_out)
        err.append(err_out)
        xsq.append(xsq_out)

    return np.array([val, err, xsq])


@au.quantity_input
def loop_change_dispersion(_p_sp_1, _p_sp_2, loop_function, iter_pts_low, iter_pts_high, iter_pts_step,
                           d_1: utils.kms, d_2: utils.kms, shift=0.3 * utils.kms, rep=1000,
                           iter_pts_are_vel=True,
                           correct_for_smooth=True,
                           correct_for_err=True, corrfunc=None):
    p_sp_1 = dc(_p_sp_1)
    p_sp_2 = dc(_p_sp_2)
    # change dispersion
    p_sp_1['dispersion'] = d_1
    p_sp_2['dispersion'] = d_2

    res = []
    if iter_pts_are_vel:
        iter_pts = np.arange(
            iter_pts_low.value, iter_pts_high.value, iter_pts_step.value)/d_1.value
        # we assume the same dispersion for both spectra
    else:
        iter_pts = np.arange(iter_pts_low, iter_pts_high, iter_pts_step)

    # note: this assumes that the dispersion does not change for the two spectra
    # fine for testing, maybe less fine for real value, but should be ok
    for i in tqdm(iter_pts):
        res.append(loop_function(
            p_sp_1, p_sp_2, shift=shift, rep=rep, n_smooth_pix_1=i, n_smooth_pix_2=i,
            correct_for_smooth=correct_for_smooth, correct_for_err=correct_for_err, corrfunc=corrfunc))
    return res, iter_pts
