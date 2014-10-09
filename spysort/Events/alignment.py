import copy
import numpy as np
from spysort.functions import convolution, cut_sgl_evt
from spysort.Events import events


class align_events(events.build_events):
    """ Alignment of spike forms after clustering using a Brute-Force method"""
    def __init__(self, data, positions, goodEvts, clusters, CSize, win=[],
                 before=14, after=30, thr=3):
        """ Performs a PCA-aided k-Means clustering and creates the proper
        indexes for further alignment of the raw data.

        **Parameters**

        data : double
            The input normalized data list

        positions : int
            The positions of spike events as they have been computed by the
            spike_detection (becareful the user has to define treat explicitly
            the size and the contents of the position array)

        clusters : double
            The clustered data

        CSize : int
            The number of the chosen clusters

        win : double
            The filtering window

        before : int
            The number of sampling point to keep before the peak

        after : int
            The number of sampling point to keep after the peak

        thr : double
            Filtering threshold value
        """
        self.win = win
        self.thr = thr
        self.before = before
        self.after = after
        self.positions = positions

        # Converts input list data to a numpy array
        self.data = np.asarray(data)
        self.goodEvts = goodEvts

        # Events instance
        events.build_events.__init__(self, self.data, self.positions, self.win,
                                     self.before, self.after)

        # k-Means clustering
        self.kmc = clusters

        # Construction of the proper cluster indices
        self.gcpos = copy.deepcopy([self.positions[self.goodEvts]
                                   [np.array(self.kmc) == i]
                                   for i in range(CSize)])

    def classify_and_align_evt(self, evt_pos, centers, abs_jitter_max=3, 
                               otherData=False, x=[]):
        """ One step of the Brute-Force method of realignment. It returns the
        name of the closest center in terms of Euclidean distance or "?" if
        none of the clusters' waveform does better than a uniformly null one,
        the new position of the event (the previous position corrected by the
        integer part of the estimated jitter), the remaining jitter.

        **Parameters**

        evt_pos : int
            A spike event time

        centers : dict
            A dictionary that contains all the necessary arrays and parameters
            in order to perform properly the classification and the alignment
            of the raw data

        abs_jitter_max : double
            The absolute maximum permitted value of the jitter
        """
        if otherData is False:
            data = self.data
        else:
            data = np.asarray(x)

        cluster_names = sorted(list(centers))
        n_sites = data.shape[0]
        centersM = np.array([centers[c_name]["center"]
                            [np.tile((-self.before <= centers[c_name]
                             ["center_idx"]).__and__(centers[c_name]
                                  ["center_idx"] <= self.after), n_sites)]
                            for c_name in cluster_names])
        evt = cut_sgl_evt(data, evt_pos, self.before, self.after)
        delta = -(centersM - evt)
        cluster_idx = np.argmin(np.sum(delta**2, axis=1))
        good_cluster_name = cluster_names[cluster_idx]
        good_cluster_idx = np.tile((-self.before <=
                                   centers[good_cluster_name]
                                   ["center_idx"]).__and__(
                                     centers[good_cluster_name]
                                    ["center_idx"] <= self.after), n_sites)
        centerD = centers[good_cluster_name]["centerD"][good_cluster_idx]
        centerD_norm2 = np.dot(centerD, centerD)
        centerDD = centers[good_cluster_name]["centerDD"][good_cluster_idx]
        centerDD_norm2 = np.dot(centerDD, centerDD)
        centerD_dot_centerDD = np.dot(centerD, centerDD)
        h = delta[cluster_idx, :]
        h_order0_norm2 = np.sum(h**2)
        h_dot_centerD = np.dot(h, centerD)
        jitter0 = h_dot_centerD/centerD_norm2
        # print jitter0
        h_order1_norm2 = np.sum((h-jitter0*centerD)**2)
        if h_order0_norm2 > h_order1_norm2:
            h_dot_centerDD = np.dot(h, centerDD)
            first = (-2. * h_dot_centerD + 2. * jitter0 *
                     (centerD_norm2 - h_dot_centerDD) + 3. * jitter0**2 *
                     centerD_dot_centerDD + jitter0**3 * centerDD_norm2)
            second = (2. * (centerD_norm2 - h_dot_centerDD) + 6. * jitter0 *
                      centerD_dot_centerDD + 3. * jitter0**2 * centerDD_norm2)
            jitter1 = jitter0 - first/second
            h_order2_norm2 = sum((h-jitter1*centerD-jitter1**2/2*centerDD)**2)
            if h_order1_norm2 <= h_order2_norm2:
                jitter1 = jitter0
        else:
            jitter1 = 0
        if np.abs(np.round(jitter1)) > 0:
            evt_pos -= int(np.round(jitter1))
            evt = cut_sgl_evt(data, evt_pos, self.before, self.after)
            h = evt - centers[good_cluster_name]["center"][good_cluster_idx]
            h_order0_norm2 = np.sum(h**2)
            h_dot_centerD = np.dot(h, centerD)
            jitter0 = h_dot_centerD/centerD_norm2
            h_order1_norm2 = np.sum((h - jitter0 * centerD)**2)
            if h_order0_norm2 > h_order1_norm2:
                h_dot_centerDD = np.dot(h, centerDD)
                first = (-2. * h_dot_centerD + 2. * jitter0 *
                         (centerD_norm2 - h_dot_centerDD) + 3. * jitter0**2 *
                         centerD_dot_centerDD + jitter0**3 * centerDD_norm2)
                second = (2. * (centerD_norm2 - h_dot_centerDD) + 6. * jitter0
                          * centerD_dot_centerDD + 3. * jitter0**2 *
                          centerDD_norm2)
                jitter1 = jitter0 - first/second
                h_order2_norm2 = np.sum((h - jitter1 * centerD - jitter1**2 /
                                         2 * centerDD)**2)
                if h_order1_norm2 <= h_order2_norm2:
                    jitter1 = jitter0
            else:
                jitter1 = 0
        if np.sum(evt**2) > np.sum((h - jitter1 * centerD - jitter1**2/2. *
                                    centerDD)**2):
            return [cluster_names[cluster_idx], evt_pos, jitter1]
        else:
            return ['?', evt_pos, jitter1]

    def get_jitter(self, evts, center, centerD, centerDD):
        """ Estimates the jitter given an event or a matrix of events where
        individual events form the rows, a median event and the first two
        derivatives of the latter.

        **Parameters**

        evts : double (array)
            The actual clean events to be realigned

        center : double (array)
            The estimate of the center (obtained from the median)

        centerD : double (array)
            The estimate of the center's derivative (obtained from the median
            of events cut on the derivative of data)

        centerDD : double (array)
            The estimate of the center's second derivative (obtained from the
            median of events cut on the second derivative of data)
        """
        centerD_norm2 = np.dot(centerD, centerD)
        centerDD_norm2 = np.dot(centerDD, centerDD)
        centerD_dot_centerDD = np.dot(centerD, centerDD)

        if evts.ndim == 1:
            evts = evts.reshape(1, len(center))

        evts = evts - center
        h_dot_centerD = np.dot(evts, centerD)
        delta0 = h_dot_centerD/centerD_norm2
        h_dot_centerDD = np.dot(evts, centerDD)
        first = (-2. * h_dot_centerD + 2. * delta0 *
                 (centerD_norm2 - h_dot_centerDD) + 3. * delta0**2 *
                 centerD_dot_centerDD + delta0**3 * centerDD_norm2)
        second = (2. * (centerD_norm2 - h_dot_centerDD) + 6. * delta0 *
                  centerD_dot_centerDD + 3. * delta0**2 * centerDD_norm2)
        return delta0 - first/second

    def mk_aligned_events(self, positions):
        """ Aligns the events of one realization. It returns a matrix of
        aligned events, a vector of spike positions giving the nearest sampling
        point to the actual peak, a vector of jitter giving the offset between
        the previous spike position and the "actual" peak position.

        **Parameters**

        positions : int (list)
            Spike times
        """
        win = np.array([1., 0., -1.])/2.
        Dx = np.apply_along_axis(convolution, 1, self.data, win)
        DDx = np.apply_along_axis(convolution, 1, Dx, win)
        evts = self.mk_events(otherPos=True, x=self.data, pos=positions)
        evtsD = self.mk_events(otherPos=True, x=Dx, pos=positions)
        evtsDD = self.mk_events(otherPos=True, x=DDx, pos=positions)
        evts_median = np.apply_along_axis(np.median, 0, evts)
        evtsD_median = np.apply_along_axis(np.median, 0, evtsD)
        evtsDD_median = np.apply_along_axis(np.median, 0, evtsDD)
        evts_jitter = self.get_jitter(evts, evts_median, evtsD_median,
                                      evtsDD_median)
        positions = positions-np.round(evts_jitter).astype('int')
        evts = self.mk_events(otherPos=True, x=self.data, pos=positions)
        evtsD = self.mk_events(otherPos=True, x=Dx, pos=positions)
        evtsDD = self.mk_events(otherPos=True, x=DDx, pos=positions)
        evts_median = np.apply_along_axis(np.median, 0, evts)
        evtsD_median = np.apply_along_axis(np.median, 0, evtsD)
        evtsDD_median = np.apply_along_axis(np.median, 0, evtsDD)
        evts_jitter = self.get_jitter(evts, evts_median, evtsD_median,
                                      evtsDD_median)
        evts -= (np.outer(evts_jitter, evtsD_median) +
                 np.outer(evts_jitter**2/2., evtsDD_median))
        return (evts, positions, evts_jitter)

    def mk_center_dictionary(self, positions):
        """ Creates a dictionary containing all the necessary information in
        order to facilitate the realignment method.

        **Parameters**

        positions : int (list)
            Spike times
        """
        win = np.array([1., 0., -1.])/2.
        dataD = np.apply_along_axis(convolution, 1, self.data, win)
        dataDD = np.apply_along_axis(convolution, 1, dataD, win)
        evts = self.mk_events(otherPos=True, x=self.data, pos=positions)
        evtsD = self.mk_events(otherPos=True, x=dataD, pos=positions)
        evtsDD = self.mk_events(otherPos=True, x=dataDD, pos=positions)
        evts_median = np.apply_along_axis(np.median, 0, evts)
        evtsD_median = np.apply_along_axis(np.median, 0, evtsD)
        evtsDD_median = np.apply_along_axis(np.median, 0, evtsDD)
        return {"center": evts_median,
                "centerD": evtsD_median,
                "centerDD": evtsDD_median,
                "centerD_norm2": np.dot(evtsD_median, evtsD_median),
                "centerDD_norm2": np.dot(evtsDD_median, evtsDD_median),
                "centerD_dot_centerDD": np.dot(evtsD_median, evtsDD_median),
                "center_idx": np.arange(-self.before, self.after+1)}

    def predict_data(self, class_pos_jitter_list, centers_dictionary,
                     nb_channels=4, data_length=300000):
        res = np.zeros((nb_channels, data_length))
        for class_pos_jitter in class_pos_jitter_list:
            cluster_name = class_pos_jitter[0]
            if cluster_name != '?':
                center = centers_dictionary[cluster_name]["center"]
                centerD = centers_dictionary[cluster_name]["centerD"]
                centerDD = centers_dictionary[cluster_name]["centerDD"]
                jitter = class_pos_jitter[2]
                pred = center + jitter*centerD + jitter**2/2*centerDD
                pred = pred.reshape((nb_channels, len(center)//nb_channels))
                idx = (centers_dictionary[cluster_name]["center_idx"] +
                       class_pos_jitter[1])
                within = np.bitwise_and(0 <= idx, idx < data_length)
                kw = idx[within]
                res[:, kw] += pred[:, within]
        return res

    def rss_for_alignment(self, delta, evt, center, centerD, centerDD):
        return sum((evt - center - delta*centerD - delta**2/2*centerDD)**2)

    def rssD_for_alignment(self, delta, evt, center, centerD, centerDD):
        h = evt - center
        return (-2*np.dot(h, centerD) + 2*delta*(np.dot(centerD, centerD) -
                np.dot(h, centerDD)) + 3*delta**2*np.dot(centerD, centerDD) +
                delta**3*np.dot(centerDD, centerDD))

    def rssDD_for_alignment(self, delta, evt, center, centerD, centerDD):
        h = evt - center
        return (2*(np.dot(centerD, centerD) - np.dot(h, centerDD)) +
                6*delta*np.dot(centerD, centerDD) +
                3*delta**2*np.dot(centerDD, centerDD))
