import copy
import numpy as np
import matplotlib.pylab as plt
from functions import mad, quantiles, curr_pos


class read_data:
    """ Fundamental methods for read, normalize and plot raw data """
    def __init__(self, Input, freq):
        """ Reads the raw data and returns the data, the data length and a
        timestamp array.

        **Parameters**

        Input : string
            A list of strings contains the full path of the input data file
            locations.

        freq : double
            The sampling frequency of the original data in order to create
            the proper timeseries.
        """
        # Check the leght of all the input channels
        if not len(np.unique(map(len,
                             map(lambda n: np.fromfile(n, dtype=np.double),
                                 Input)))):
            print 'Data dimensions mismatch'

        # Data length
        self.data_len = np.unique(map(len, map(
                                  lambda n: np.fromfile(n, dtype=np.double),
                                  Input)))[0]

        # Loading the raw data from different files
        self.data = []
        for i, v in enumerate(map(lambda n: np.fromfile(n, dtype=np.double),
                                  Input)):
            self.data.append(v)

        # Create timestamps in order to facilitate plotting the data
        self.timebase = np.arange(0, self.data_len)/freq
        # Create a super-list containing the data and the timestamps
        self.timeseries = copy.copy([self.data, list(self.timebase)])

    def renormalization(self):
        """ Data renormalization (divise by mad) """
        mad_values = np.apply_along_axis(mad, 1, self.data)
        median_values = np.apply_along_axis(np.median, 1, self.data)
        return copy.deepcopy([((self.data[i] - median_values[i])/mad_values[i])
                             for i in range(len(self.data))])

    def subseting(self, begin=0, end=1):
        """ Checks if a subset of the input data is continuous.

            **Parameters**

            begin : int
                The first element of data points to be selected

            end : int
                The last element of data points
        """
        Dx = [np.sign(np.abs(np.diff(self.data[i][begin:end], 1))) for i in
              xrange(len(self.data))]
        check = [all(Dx[i] == 1) for i in xrange(len(Dx))]
        if all(check) is True:
            return copy.deepcopy([self.data[i][begin:end] for i in
                                  range(np.abs(begin-end))])
        else:
            print 'The resulting array contains not continuous data'

    def select_channels(self, channels, begin=0, end=1):
        """ Selects a subset of channels and checks all the necessary
            constraints.

            **Parameters**

            channels : int
                The number of channels to be selected

            begin : int
                The first element of data points to be selected

            end : int
                The last element of data points
        """
        stop = len(self.data)

        if len(channels) > stop:
            print 'Requested number of channels exceeded number of available'
            print 'channels!'
        elif any([channels[i] > stop for i in range(len(channels))]):
            print 'Requested number of channels exceeded number of available'
            print 'channels!'
        elif any([channels[i] < 0 for i in range(len(channels))]):
            print 'No negative channels!'
        else:
            if (begin == 0) & (end == self.data_len):
                return copy.deepcopy([self.data[i] for i in channels])
            else:
                return copy.deepcopy([self.subseting(begin, end)[i] for i in
                                     range(self.data_len)])

    def summary(self):
        """ Prints the mad, the median and the quantiles of the data """
        print 'MAD :'
        print np.apply_along_axis(mad, 1, self.data)
        print 'Median :'
        print np.apply_along_axis(np.median, 1, self.data)
        print 'Quantiles: '
        print np.apply_along_axis(quantiles, 1, self.data)

    def plot_data(self, x, figsize=(9, 8), save=False, figname='WholeRawData',
                  figtype='png'):
        """ Plots the data. Can interact with the user by supporting scrolling
            and selective printing of data segments.

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
        x = np.asarray(x)
        step = 2000

        def key_event(e):
            global curr_pos, fname
            if e.key == 'right':
                curr_pos = curr_pos + step
            elif e.key == 'left':
                curr_pos = curr_pos - step
            elif e.key == 'ctrl+p':
                fname = raw_input('Please enter the name of this frame: ')
                if figtype == 'pdf':
                    plt.savefig(fname+'.pdf', dpi=90)
                else:
                    plt.savefig(fname+'.png')
            else:
                return
            ax.cla()
            A = 0
            for i in range(x.shape[0]):
                ax.plot(self.timebase[curr_pos:curr_pos+step],
                        x[i, curr_pos:curr_pos+step] - A, 'k')
                A += 15
            ax.set_yticks([])
            fig.canvas.draw()

        fig = plt.figure(figsize=figsize, frameon=False)
        fig.canvas.mpl_connect('key_press_event', key_event)
        ax = fig.add_subplot(111)
        B = 0
        for i in range(x.shape[0]):
            ax.plot(self.timebase[curr_pos:curr_pos+step],
                    x[i, curr_pos:curr_pos+step] - B, 'k')
            B += 15
        ax.set_xlabel('Time (s)')
        ax.set_xlim([self.timebase[curr_pos:curr_pos+step].min(),
                     self.timebase[curr_pos:curr_pos+step].max()])
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        if save:
            if figtype == 'pdf':
                plt.savefig(figname+'pdf', dpi=90)
            else:
                plt.savefig(figname+'png')
