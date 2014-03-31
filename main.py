import numpy as np
# from guppy import hpy
# from events import events
from read_data import read_data
import matplotlib.pylab as plt
# from functions import quantiles
# from alignment import alignment
from clustering import pca_clustering
from spike_detection import spike_detection

if __name__ == '__main__':
    folder = '/Users/gdetorak/Documents/SUPELEC/Software/Data/Locust/'
    IFiles = [folder+'Locust_1.dat', folder+'Locust_2.dat',
              folder+'Locust_3.dat', folder+'Locust_4.dat']

    # h = hpy()
    freq = 1.5e4
    win = np.array([1., 1., 1., 1., 1.])/5.
    # Test of rawData class
    r_data = read_data(IFiles, freq)
    # r_data.summary()
    # res = r_data.select_channels([1, 2], 0, r_data.data_len)
    r_data.timeseries[0] = r_data.renormalization()

    # Test of spike_detection class
    spikes = spike_detection(r_data.timeseries)
    filtered = spikes.filtering(4.0, win)
    sp0 = spikes.peaks(filtered, kind='aggregate')
    positions = sp0[sp0 <= r_data.data_len/2.]
    # spikes.plot_filtered_data(spikes.data, filtered, 4.)
    # spikes.plot_peaks(spikes.data, sp0)

    # Test of events class
    # evts = events(r_data.data, positions, win, before=14, after=30)
    # evtsE = evts.mk_events()
    # noise = evts.mk_noise()
    # evts.plot_mad_median(evtsE)
    # evts.plot_events(evtsE)

    # Test PCA and KMeans clustering
    clusters = pca_clustering(r_data.timeseries[0], positions, win, thr=8,
                              before=14, after=30)
    # print clusters.pca_variance(10)
    # clusters.plot_mean_pca()
    # clusters.plot_pca_projections()
    # kmeans_clusters = clusters.KMeans(10)
    # gmm_clusters = clusters.GMM(10, 'diag')
    # tree = clusters.bagged_clustering(10, 100, 30)
    # clusters.plot_clusters(tree)

    # align = alignment(IFiles, freq, win, before=14, after=30, CSize=10)
    # evtsE_noj = [align.mk_aligned_events(align.gcpos[i])
    #              for i in range(10)]

    # centers = {"Cluster " + str(i): align.mk_center_dictionary(evtsE_noj[i][1])
    #            for i in range(10)}

    # round0 = [align.classify_and_align_evt(align.positions[i], centers)
    #           for i in range(len(align.positions))]

    # print len([x[1] for x in round0 if x[0] == '?'])
    # print h.heap()
    plt.show()
