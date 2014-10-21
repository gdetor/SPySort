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

        **Returns**

        The median absolute deviation of input signal x.
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

        **Returns**

        A vector containing all the computed quantilies for the input signal x.
    """
    return st.mstats.mquantiles(x, prob=prob)


def convolution(x, window):
    """ A scipy fftconvolution wrapper function.

        **Parameters**

        x : double
            Data array

        window : double
            Filter window

        **Returns**

        The convolution of the input signal with a predefined window (win).
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

        **Returns**
        A numpy array containing the data for which the |X-median(X)|/mad(X) <
        thr.
    """
    return np.ndarray.all(np.abs((x - median)/mad) < thr)


def good_evts_fct(x, thr=3):
    """ Clean the spike events times.

        **Parameters**

        x : double
            Data array

        thr : double
            Threshold of filtering

        **Returns**
        A vector containing all the detected good events.
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
            The index (location) of the (peak of) the event.

        before : int
            How many points should be within the cut before the reference
            index / time given by position.

        after : int
            How many points should be within the cut after the reference
            index / time given by position.

        **Returns**
        A vector with the cuts on the different recording sites glued one after
        the other.
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
