import copy
import numpy as np
from functions import mad, cut_sgl_evt
import matplotlib.pylab as plt
from spike_detection import spike_detection


class events(spike_detection):
    """ Create all the spike events """
    def __init__(self, data, positions, win, before=14, after=30):
        """ Reads the raw data applies a filtering and detects all the possible
        events

        **Parameters**

        Input : string
            A list of strings contains the full path of the input data file
            locations.

        freq : double
            The sampling frequency of the original data in order to create
            the proper timeseries.


        win : double
            A numpy array contains the filter window that is used in filtering

        before : int
            The number of points before of an peak

        after : int
            The number of points after a peak (both parameters, before and
            after, define the size of an event segment)

        """
        self.before = before
        self.after = after

        # Convert the data list to a numpy array
        self.data = np.asarray(data)
        # Data filtering
        self.positions = positions

    def mk_events(self, otherPos=False, x=[], pos=[]):
        """ Constructs a list of all the events from the input data.

            **Parameters**

            x : double (array)
                Input data

            otherPos : boolean
                It defines if the user wants to use a different spike time
                list than the one provided by the constructor of the class

            pos : int (list)
                The new list of event positions defined by the user
        """
        if otherPos is False:
            res = np.zeros((len(self.positions),
                           (self.before+self.after+1) * self.data.shape[0]))
            for i, p in enumerate(self.positions):
                res[i, :] = cut_sgl_evt(self.data, p, self.before, self.after)
            return copy.deepcopy(res)
        else:
            res = np.zeros((len(pos),
                           (self.before+self.after+1) * x.shape[0]))
        for i, p in enumerate(pos):
            res[i, :] = cut_sgl_evt(x, p, self.before, self.after)
        return copy.deepcopy(res)

    def mk_noise(self, otherPos=False, x=[], safety_factor=2, size=2000):
        """ Computes the noise events

            **Parameters**

            x : double
                Input data

            safety_factor : int
                A number by which the cut length is multiplied and which
                sets the minimal distance between the reference times

            size : int
                The maximal number of noise events one wants to cut
        """
        if otherPos is False:
            x = self.data
        else:
            x = x

        sl = self.before + self.after + 1   # cut length
        i1 = np.diff(self.positions)        # inter-event intervals
        minimal_length = round(sl*safety_factor)
        # Get next the number of noise sweeps that can be
        # cut between each detected event with a safety factor
        nb_i = (i1 - minimal_length)//sl
        # Get the number of noise sweeps that are going to be cut
        nb_possible = min(size, sum(nb_i[nb_i > 0]))
        res = np.zeros((nb_possible, sl * x.shape[0]))
        # Create next a list containing the indices of the inter event
        # intervals that are long enough
        idx_l = [i for i in range(len(i1)) if nb_i[i] > 0]
        # Make next an index running over the inter event intervals
        # from which at least one noise cut can be made
        interval_idx = 0
        # noise_positions = np.zeros(nb_possible,dtype=numpy.int)
        n_idx = 0
        while n_idx < nb_possible:
            within_idx = 0  # an index of the noise cut with a long
                            # enough interval
            i_pos = int(self.positions[idx_l[interval_idx]] + minimal_length)
            # Variable defined next contains the number of noise cuts
            # that can be made from the "currently" considered long-enough
            # inter event interval
            n_at_interval_idx = nb_i[idx_l[interval_idx]]
            while within_idx < n_at_interval_idx and n_idx < nb_possible:
                res[n_idx, :] = cut_sgl_evt(x, i_pos, self.before, self.after)
                ## noise_positions[n_idx] = i_pos
                n_idx += 1
                i_pos += sl
                within_idx += 1
            interval_idx += 1
        return res

    def sieve(self, func, x, *args):
        """ It sieves the events in order to get the clean ones.

        **Parameters**

        func : function
            It's a user defined functions, that defines the cleaning method
            of the events

        *args : arguments (double)
            It's actually the threshold of the sieving.
        """
        tmp = func(x, *args)
        return tmp

    def plot_mad_median(self, events, figsize=(5, 5), save=False,
                        figname='mam-median-evts', figtype='png'):
        """ Plots the median and the medan absolute value of the input
            array x

            **Parameters**

            x : double
                A list of the input data

            figsize : float
                A tuple of the sizes of the figure

            save : boolean
                Indicates if the figure will be saved

            figname : string
                The name of the saved figure

            figtype : string
                The type of the saved figure (supports png and pdf)
        """
        events_median = np.apply_along_axis(np.median, 0, events)
        events_mad = np.apply_along_axis(mad, 0, events)

        plt.figure(figsize=figsize)
        plt.plot(events_median, color='red', lw=2)
        plt.axhline(y=0, color='black')
        for i in np.arange(0, 400, 100):
            plt.axvline(x=i, c='k', lw=2)
        for i in np.arange(0, 400, 10):
            plt.axvline(x=i, c='grey')
        plt.plot(events_median, 'r', lw=2)
        plt.plot(events_mad, 'b', lw=2)
        if save:
            if figtype == 'pdf':
                plt.savefig(figname+'pdf', dpi=90)
            else:
                plt.savefig(figname+'png')

    def plot_events(self, x, r=(0, 200), figsize=(5, 5), save=False,
                    figname='mam-median-evts', figtype='png'):
        """ Plots all the computed events

            **Parameters**

            x : double
                A list of the input data

            r : int
                A tuple contains the range of the plotting data

            figsize : float
                A tuple of the sizes of the figure

            save : boolean
                Indicates if the figure will be saved

            figname : string
                The name of the saved figure

            figtype : string
                The type of the saved figure (supports png and pdf)
        """
        x = np.asarray(x)
        plt.figure(figsize=figsize)
        for i in range(r[0], r[1]):
            plt.plot(x[i, :], 'k', lw=0.1)
            plt.plot(np.apply_along_axis(np.median, 0, x[r[0]:r[1], :]),
                     'r', lw=1)
            plt.plot(np.apply_along_axis(mad, 0, x[r[0]:r[1], :]), 'b',
                     lw=1)
            plt.axvspan(45, 89, fc='grey', alpha=0.5, edgecolor='none')
            plt.axvspan(135, 179, fc='grey', alpha=0.5, edgecolor='none')
