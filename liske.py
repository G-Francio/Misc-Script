import numpy as np

import utilities as utils
import spec as gspec


# all the formulas I need to use
def vel_err_per_pix(F1_i, F2_i, F1_ip1, F2_ip1, eF1_i, eF2_i, eF1_ip1, eF2_ip1, dv):
    """
    Computes the velocity uncertainty, squared, per each pixel i.
    """
    grad = gradient_i(F1_i, F2_i, F1_ip1, F2_ip1, eF1_i, eF2_i, dv)
    eGrad2 = e_gradient_i(eF1_i, eF2_i, eF1_ip1, eF2_ip1)
    prefact = (1 / (grad))**2
    # / (aconst.c.value / v)**2 -> I think this should not be there
    fact = (F2_i - F1_i)**2 / (grad)**2
    return prefact * (eF1_i**2 + eF2_i**2 + fact * eGrad2)


def liske_unc(unc_per_pix):  # checked
    return 1 / np.sum(unc_per_pix**(-1))


def gradient_i(F1_i, F2_i, F1_ip1, F2_ip1, eF1_i, eF2_i, dv):  # checked
    """
    Computes dS_i/dv (eq. 11 paper)
    """
    num = (F1_ip1 - F1_i)/eF1_i**2 + (F2_ip1 - F2_i)/eF2_i**2
    den = (1/eF1_i**2 + 1/eF2_i**2)
    return num / den / dv


def e_gradient_i(eF1_i, eF2_i, eF1_ip1, eF2_ip1):
    """
    Computes the uncertainty on the gradient squared (eq. 12 paper)
    """
    prefact = (1/eF1_i**2 + 1/eF2_i**2)**(-2)
    fact = (eF1_ip1**2 + eF1_i**2)/eF1_i**2 + (eF2_ip1**2 + eF2_i**2)/eF2_i**2
    return prefact * fact


def vel_err_per_pix_eq_14(F1_i, F2_i, F1_ip1, F2_ip1, eF1_i, eF2_i, eF1_ip1, eF2_ip1, dv):
    """
    Eq. 14 from the paper, even if I have some doubts with respect to the absence of the v everywhere.
    """
    pref = gradient_i(F1_i, F2_i, F1_ip1, F2_ip1, eF1_i, eF2_i, dv)
    z = (F2_i - F1_i)**2 / ((F1_ip1 - F1_i)/eF1_i**2 + (F2_ip1 - F2_i)/eF2_i **
                            2)**2 * (eF1_ip1**2/eF1_i**4 + eF2_ip1**2/eF2_i**4 + 1/eF1_i**2 + 1/eF2_i**2)
    # if you look at z, it is different than the paper - but this is at the very least dimensionally correct
    return 1 / pref**2 * (eF1_i**2 + eF2_i**2 + z)


def _flux_err_no_smooth(sp_1, sp_2):
    f_i_1 = sp_1.y.value[:-1]
    f_i_2 = sp_2.y.value[:-1]
    f_ip1_1 = sp_1.y.value[1:]
    f_ip1_2 = sp_2.y.value[1:]
    e_f_i_1 = sp_1.dy.value[:-1]
    e_f_i_2 = sp_2.dy.value[:-1]
    e_f_ip1_1 = sp_1.dy.value[1:]
    e_f_ip1_2 = sp_2.dy.value[1:]
    return f_i_1, f_i_2, f_ip1_1, f_ip1_2, e_f_i_1, e_f_i_2, e_f_ip1_1, e_f_ip1_2


def _flux_err_smooth(sp_1, sp_2):
    f_i_1 = sp_1.y_smooth.value[:-1]
    f_i_2 = sp_2.y_smooth.value[:-1]
    f_ip1_1 = sp_1.y_smooth.value[1:]
    f_ip1_2 = sp_2.y_smooth.value[1:]
    e_f_i_1 = sp_1.dy_corr.value[:-1]
    e_f_i_2 = sp_2.dy_corr.value[:-1]
    e_f_ip1_1 = sp_1.dy_corr.value[1:]
    e_f_ip1_2 = sp_2.dy_corr.value[1:]
    return f_i_1, f_i_2, f_ip1_1, f_ip1_2, e_f_i_1, e_f_i_2, e_f_ip1_1, e_f_ip1_2


def _flux_err_spline(sp_1, sp_2):
    f_i_1 = sp_1.y_smooth.value[:-1]
    f_i_2 = sp_2.y_spline(sp_1.x.value)[:-1]
    f_ip1_1 = sp_1.y_smooth.value[1:]
    f_ip1_2 = sp_2.y_spline(sp_1.x.value)[1:]
    e_f_i_1 = sp_1.dy_corr.value[:-1]
    e_f_i_2 = sp_2.dy_spline_corr(sp_1.x.value)[:-1]
    e_f_ip1_1 = sp_1.dy_corr.value[1:]
    e_f_ip1_2 = sp_2.dy_spline_corr(sp_1.x.value)[1:]
    return f_i_1, f_i_2, f_ip1_1, f_ip1_2, e_f_i_1, e_f_i_2, e_f_ip1_1, e_f_ip1_2


# Thanks ChatGPT for the cleaner function
def compute_liske_uncertainty(sp_1, sp_2, which="", xi_sigma=np.nan, xi_fwhm=np.nan):
    if which not in ["", "smooth", "spline"]:
        raise ValueError(
            "Please choose a `which` between '', 'smooth', 'spline'.")
    dv = sp_1.x.value[1:] - sp_1.x.value[:-1]
    if which == "":
        out = vel_err_per_pix_eq_14(*_flux_err_no_smooth(sp_1, sp_2), dv)
    elif which == "spline":
        out = vel_err_per_pix_eq_14(*_flux_err_spline(sp_1, sp_2), dv)
    else:
        out = vel_err_per_pix_eq_14(*_flux_err_smooth(sp_1, sp_2), dv)
    if which == '':
        return np.sqrt(liske_unc(out))
    else:
        out = out / (xi_sigma * xi_fwhm)**2
        return np.sqrt(liske_unc(out))


def liske_gauss_noise(x, mu, sigma):
    n = x.value.shape[0]
    noise_liske = np.random.normal(mu, sigma, n)
    noise_err_liske = np.ones(n) * sigma
    return gspec.mock_spec(wave=x, flux=noise_liske * utils.adm, err=noise_err_liske * utils.adm)


def generate_tracker_pts(sp_1, sp_2, n_rep, n_smooth_pix_1, n_smooth_pix_2):
    tracker_points = []
    for _ in range(n_rep):
        sp_tracker_1 = liske_gauss_noise(
            sp_1.x, 1., np.median(sp_1.dy.value))
        sp_tracker_2 = liske_gauss_noise(
            sp_2.x, 1., np.median(sp_2.dy.value))

        sp_tracker_1.smooth(n_smooth_pix_1)
        sp_tracker_2.spline(n_smooth_pix_2)

        tracker_points.append(
            compute_liske_uncertainty(sp_tracker_1, sp_tracker_2, which=''))

    return np.mean(tracker_points), np.std(tracker_points)


def dv_i(F1_im1, F2_im1, F1_i, F2_i, F1_ip1, F2_ip1, eF1_i, eF2_i, dv):
    # this computes the shift in km/s, in principle
    return (F2_i - F1_i)/gradient_i(F1_im1, F2_im1, F1_ip1, F2_ip1, eF1_i, eF2_i, dv)


def dv_sum(sp_1, sp_2):
    # for now assume km/s
    # also note that this is computed for a slightly smaller range
    # could get around this by passing an array that is one pixel larger on
    # both sides, but I don't think it's going to matter much
    out = []
    for i in range(0, len(sp_1.x) - 1):
        dv = (sp_1.x[i + 1] - sp_1.x[i]).value
        out.append(dv_i(sp_1.y_smooth[i], sp_2.y_smooth[i], sp_1.y_smooth[i], sp_2.y_smooth[i],
                   sp_1.y_smooth[i+1], sp_2.y_smooth[i+1], sp_1.dy[i], sp_2.dy[i], dv))
    return out


def weighted_dv(unw_dv, weight):
    return np.sum(unw_dv * 1/weight**2) / np.sum(1/weight**2)


def bouchy(sp_1, sp_2, which=''):
    if which not in ["", "smooth", "spline"]:
        raise ValueError(
            "Please choose a `which` between '', 'smooth', 'spline'.")
    unw_dv = dv_sum(sp_1, sp_2)
    dv = sp_1.x.value[1:] - sp_1.x.value[:-1]
    if which == "":
        weight = vel_err_per_pix_eq_14(*_flux_err_no_smooth(sp_1, sp_2), dv)
    elif which == "spline":
        weight = vel_err_per_pix_eq_14(*_flux_err_spline(sp_1, sp_2), dv)
    else:
        weight = vel_err_per_pix_eq_14(*_flux_err_smooth(sp_1, sp_2), dv)
    return weighted_dv(unw_dv, weight), compute_liske_uncertainty(sp_1, sp_2)
