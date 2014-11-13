import copy
import collections
import numpy as np
import matplotlib.pylab as plt
from scipy.stats.mstats import mquantiles
from spysort.functions import mad, quantiles, curr_pos
# from functions import mad, quantiles, curr_pos

Data_timeseries = collections.namedtuple('Data_timeseries', 'Data Timestamps')


class read_data(object):
    """ Fundamental methods for read, normalize and plot raw data """

    def __init__(self, input_path, sampling_freq):
        """ Reads the raw data and returns the data, the data length and a
        timestamp array.

        Args:
            Input (list of str) : Contains the full path of the input data file
                                  locations.

            freq (double) : The sampling frequency of the original data.

        Returns:
            data_len (int) : The size of the input data.

            data (list) : The input data.

            timebase (int array) : Timestamps for facilitating plotting.

            timeseries (list) : A super-list that contains the raw data and the
                                timestamps.

        Raises:

        """
        self._set_inpath(input_path)
        self._set_freq(sampling_freq)

        # Check the leght of all the input channels
        if not len(np.unique(map(len,
                             map(lambda n: np.fromfile(n, dtype=np.double),
                                 self.inpath)))):
            print 'Data dimensions mismatch'

        # Data length
        self.data_len = np.unique(map(len, map(
                                  lambda n: np.fromfile(n, dtype=np.double),
                                  self.inpath)))[0]

        # Loading the raw data from different files
        self.data = []
        for i, v in enumerate(map(lambda n: np.fromfile(n, dtype=np.double),
                                  self.inpath)):
            self.data.append(v)

        self.timebase = np.arange(0, self.data_len)/self.freq
        self.timeseries = copy.copy([self.data, list(self.timebase)])
        Data_timeseries(self.data, self.timebase)

    def renormalization(self):
        """ Data renormalization (divise by mad) """
        mad_values = np.apply_along_axis(mad, 1, self.data)
        median_values = np.apply_along_axis(np.median, 1, self.data)
        return copy.deepcopy([((self.data[i] - median_values[i])/mad_values[i])
                             for i in range(len(self.data))])

    def subseting(self, begin=0, end=1):
        """ Checks if a subset of the input data is continuous.
        Args:

        Kwargs:
            begin (int) : The first element of data points to be selected.

            end (int) : The last element of data points.

        Returns:
            A copy of a list that contains the data within the range
            [begin,end].

        Raises:
        """
        Dx = [np.sign(np.abs(np.diff(self.data[i][begin:end], 1))) for i in
              xrange(len(self.data))]
        check = [all(Dx[i] == 1) for i in xrange(len(Dx))]
        if all(check) is True:
            return copy.deepcopy([self.data[i][begin:end] for i in
                                  range(np.abs(begin-end))])
        else:
            print 'The resulting array contains not continuous data'

    def selectChannels(self, channels, begin=0, end=1):
        """ Selects a subset of channels and checks all the necessary
            constraints.

        Args:
            channels (int) : The number of channels to be selected.

        Kwargs:
            begin (int) : The first element of data points to be selected.

            end (int) : The last element of data points.

        Returns:
            A list that contains data from a specified number of channels and
            within a predefined range [begin, end].

        Raises:
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

    def fiveNumbers(self):
        """ Returns the five numbers, the minimum, the first quartile, the
            median, the third quartile and the maximum """
        np.set_printoptions(precision=3)
        return [mquantiles(i, prob=[0, 0.25, 0.5, 0.75, 1]) for i in self.data]

    def checkStdDiv(self):
        """ Checks if the raw data have been divided by any std """
        return [np.std(i) for i in self.data]

    def discretStepAmpl(self):
        """ Returns the discretization step amplitute of raw data """
        return [np.min(np.diff(np.sort(np.unique(i)))) for i in self.data]

    def checkMad(self):
        """ Checks if the data have been properly renormalized and MAD works
            fine """
        from scipy.stats import norm
        probs = np.arange(0.01, 0.99, 0.001)
        dataQ = map(lambda x:
                    mquantiles(x, prob=probs), self.data)
        dataQsd = map(lambda x:
                      mquantiles(x/np.std(x), prob=probs), self.data)
        qq = norm.ppf(probs)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100), c='grey')
        colors = ['black', 'orange', 'blue', 'red']
        for i, y in enumerate(dataQ):
            ax.plot(qq, y, color=colors[i])

        for i, y in enumerate(dataQsd):
            ax.plot(qq, y, color=colors[i], linestyle="dashed")

        ax.set_xlabel('Normal quantiles')
        ax.set_ylabel('Empirical quantiles')

    def plotData(self, x, figsize=(9, 8), save=False, figname='WholeRawData',
                 figtype='png'):
        """ Plots the data. Can interact with the user by supporting scrolling
            and selective printing of data segments.

        Args:
            x (list) : A list of the input data

        Kwargs:
            figsize (tuple of floats) : Size of the figure.

            save (boolean) : Indicates if the figure will be saved.

            figname (str) : The name of the file in which the figure will be
                            saved.

            figtype (str) : The type of the file in which the figure will be
                            saved (supports png and pdf).

        Returns:

        Raises:
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
        idx_ = ax.get_xticks([])
        tmp_ = [str(i/10000.0) for i in idx_]
        ax.set_xticklabels(tmp_)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        if save:
            if figtype == 'pdf':
                plt.savefig(figname+'pdf', dpi=90)
            else:
                plt.savefig(figname+'png')

    """ Accessors """
    def _set_inpath(self, input_path):
        """ Accessor for input path.

        Args:
            input_path (list) : Full input paths where the input files are
                                located.
        Returns:
            inpath (list) : Same as input_path but this is a class attribute.

        Raises:
            ValueError if the list contains no strings.
        """
        if all(isinstance(item, basestring) for item in input_path) is False:
            raise ValueError('Input is not a string!')
        self.inpath = input_path

    def _set_freq(self, sampling_freq):
        """ Accessor for sampling frequency.

        Args:
            sampling_freq (double) : Sampling frequency of the input raw data.

        Returns:
            freq (double) : Same as sampling_freq, but this is a class
                            attribute.
        Raises:
            ValueError if the frequency is not a double/float.
        """
        if isinstance(sampling_freq, np.float) is False:
            raise ValueError("freq is not a float!")
        self.freq = sampling_freq

    """ Properties """
    input_path = property(_set_inpath,
                          doc='The full path where input files are.'
                          )

    sampling_freq = property(_set_freq,
                             doc="The sampling frequency of the input data."
                             )
