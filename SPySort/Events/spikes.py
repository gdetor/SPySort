import copy
import numpy as np
import matplotlib.pylab as plt
# from functions import convolution, mad, curr_pos, cp
from SPySort.functions import mad, convolution, curr_pos, cp


class spike_detection():
    """ Performs the spike detection over all the raw input data """
    def __init__(self, data):
        """ Reads the input normalized data.

            **Parameters**

            data : double
                A list of size two. The first element contains all the
                normalized data and the second one the timestamps
        """
        self.data = data[0]

        # Check if the data are normalized
        if all(np.not_equal(np.apply_along_axis(mad, 1, self.data), 1)):
            print 'Critical Error: Data are not normalized'

        self.timebase = np.asarray(data[1])

    def filtering(self, threshold, window):
        """ Filters the raw data using a threshold value.

            **Parameters**

            threshold : double
                Filtering threshold value

            window : double (array)
                The actual filter (boxcar, butter, wiener, etc)
        """
        filtered_data = np.apply_along_axis(convolution, 1, self.data,
                                            window)
        mad_value = np.apply_along_axis(mad, 1, filtered_data)
        filtered_data = [filtered_data[i] / mad_value[i] for i in
                         range(len(self.data))]
        for i in range(len(self.data)):
            filtered_data[i][filtered_data[i] < threshold] = 0
        return filtered_data

    def peaks(self, x, minimalDist=15, notZero=1e-3, kind='aggregate'):
        """ Detects the peaks over filtered data

            **Parameters**
            x : double
                Input filtered data

            minimalDist : int
                A number that defines how many sampling points will be kept.
                (only putative maxima that are farther apart than minimalDist
                 sampling points are kept)

            notZero : double
                Definition of zero

            kind : string
                aggregate : performs the detection on the sum of the input data
                othr : performs the detection on each channel separately
        """
        win = np.array([1, 0, -1])/2.
        x = np.asarray(x)
        if kind == 'aggregate':
            tmp_x = x.sum(0)
            dx = convolution(tmp_x, win)
            dx[np.abs(dx) < notZero] = 0
            dx = np.diff(np.sign(dx))
            pos = np.arange(len(dx))[dx < 0]
            return pos[:-1][np.diff(pos) > minimalDist]
        elif kind == 'full':
            pos = []
            dx = np.apply_along_axis(convolution, 1, x, win)
            dx[:, np.abs(dx[:, ...]) < notZero] = 0
            dx = np.diff(np.sign(dx))
            for i in range(dx.shape[0]):
                tmp = np.arange(len(dx[i]))[dx[i, ...] < 0]
                pos.append(tmp[:-1][np.diff(tmp) > minimalDist])
            return copy.copy(pos)
        else:
            print 'You have to choose either the aggregate or the full mode'

    def plot_filtered_data(self, x, y, thrs, figsize=(9, 8), save=False,
                           figname='candidate_peaks', figtype='png'):
        """ Plots the raw data vs the filtered ones according to the rawData
            plot method.

            **Parameters**

            x : double
                A list of the input data

            y : int
                A list of the filtered data

            thrs : double
                The filtering threshold value

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
        y = np.asarray(y)
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
                ax.plot(self.timebase[curr_pos:curr_pos+step],
                        y[i, curr_pos:curr_pos+step] - A, 'r')
                ax.axhline(thrs - A, c='k', ls='--')
                A += 15
            ax.set_yticks([])
            fig.canvas.draw()

        B = 0
        fig = plt.figure(figsize=figsize, frameon=False)
        fig.canvas.mpl_connect('key_press_event', key_event)
        ax = fig.add_subplot(111)
        for i in range(x.shape[0]):
            ax.plot(self.timebase[curr_pos:curr_pos+step],
                    x[i, curr_pos:curr_pos+step] - B, 'k')
            ax.plot(self.timebase[curr_pos:curr_pos+step],
                    y[i, curr_pos:curr_pos+step] - B, 'r')
            ax.axhline(thrs - B, c='k', ls='--')
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

    def plot_peaks(self, x, y, figsize=(9, 8), save=False, figname='peaks',
                   figtype='png'):
        """ Plots the filtered data (in black) and the detected peaks
            (in red).

            **Parameters**

            x : double
                A list of the input data

            y : int
                A list of peak positions

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
        y = np.asarray(y, dtype='int')
        step = 2000

        def key_event(e):
            global cp, fname
            if e.key == 'right':
                cp = cp + step
            elif e.key == 'left':
                cp = cp - step
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
                ax.plot(self.timebase[cp:cp+step], x[i, cp:cp+step] - A, 'k')
                idx = (y > cp) & (y < cp+step)
                ax.plot(self.timebase[y[idx]], x[i, y[idx]]
                        - A, 'ro')
                A += 15
            ax.set_yticks([])
            fig.canvas.draw()

        B = 0
        fig = plt.figure(figsize=figsize, frameon=False)
        fig.canvas.mpl_connect('key_press_event', key_event)
        ax = fig.add_subplot(111)
        for i in range(x.shape[0]):
            ax.plot(self.timebase[cp:cp+step], x[i, cp:cp+step] - B, 'k')
            idx = (y > cp) & (y < cp+step)
            ax.plot(self.timebase[y[idx]], x[i, y[idx]] - B, 'ro')
            B += 15
        ax.set_xlabel('Time (s)')
        idx_ = ax.get_xticks([])
        tmp_ = [str(i/10000.0) for i in idx_]
        ax.set_xticklabels(tmp_)

        ax.set_xlim([self.timebase[cp:cp+step].min(),
                     self.timebase[cp:cp+step].max()])
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
