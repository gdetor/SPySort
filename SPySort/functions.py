import copy
import numpy as np
from scipy import signal
import scipy.stats as st

curr_pos, cp, fname = 0, 0, 'frame'


def mad(x):
    """ Returns the median absolute deviation.

        **Parameters**

        x : double
            Data array
    """
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def quantiles(x, prob=[0, 0.25, 0.5, 0.75, 1]):
    """ Computes the five quantiles of the input vector,
        prob is a list of quantiles to compute.

        **Parameters**

        x : double
            Data array

        prob : double
            List of quantiles to compute
    """
    return st.mstats.mquantiles(x, prob=prob)


def convolution(x, window):
    """ A scipy fftconvolution wrapper function.

        **Parameters**

        x : double
            Data array

        window : double
            Filter window
    """
    return signal.fftconvolve(x, window, 'same')


def f(x, median, mad, thr):
    """ Auxiliary function, used by good_evts_fct.

        **Parameters**

        x : double
            Data array

        median : double
            Array contains median values

        mad : double
            Array contains mad values

        thr : double
            Filtering threshold
    """
    return np.ndarray.all(np.abs((x - median)/mad) < thr)


def good_evts_fct(x, thr=3):
    """ Clean the spike events times.

        **Parameters**

        x : double
            Data array

        thr : double
            Threshold of filtering
    """
    samp_median = np.apply_along_axis(np.median, 0, x)
    samp_mad = np.apply_along_axis(mad, 0, x)
    above = samp_median > 0

    samp_r = copy.copy(x)

    for i in range(len(x)):
        samp_r[i][above] = 0

    samp_median[above] = 0
    res = np.apply_along_axis(f, 1, samp_r, samp_median,
                              samp_mad, thr)
    return res


def cut_sgl_evt(x, position, before, after):
    """ Draw a singles event from the input data.

        **Parameters**

        x : double (array)
            Input data

        position : int
            The spike event time

        before : int
            The number of sampling point to keep before the peak

        after : int
            The number of sampling point to keep after the peak
    """
    ns = x.shape[0]             # Number of recording sites
    dl = x.shape[1]             # Number of sampling points
    cl = before + after + 1     # The length of the cut
    cs = cl * ns                # The 'size' of a cut
    cut = np.zeros((ns, cl))
    idx = np.arange(-before, after + 1)
    keep = idx + position
    within = np.bitwise_and(0 <= keep, keep < dl)
    kw = keep[within]
    cut[:, within] = x[:, kw].copy()
    return cut.reshape(cs)
