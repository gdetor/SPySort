import numpy as np
import matplotlib.pylab as plt
# from guppy import hpy

from ReadData import data
from Events import spikes
from Events import events
from Events import clusters
from Events import alignment

if __name__ == '__main__':
    # Define the raw data location
    folder = '/Users/gdetorak/Documents/SUPELEC/Software/Data/Locust/'
    IFiles = [folder+'Locust_1.dat', folder+'Locust_2.dat',
              folder+'Locust_3.dat', folder+'Locust_4.dat']

    # Data analysis parameters
    freq = 1.5e4    # Sampling frequency
    win = np.array([1., 1., 1., 1., 1.])/5.  # Boxcar filter

    # Test of rawData class
    r_data = data.read_data(IFiles, freq)  # Read raw data
    # r_data.summary()                       # Prints a data summary

    # res = r_data.select_channels([1, 2], 0, r_data.data_len)  # Selects chan

    r_data.timeseries[0] = r_data.renormalization()  # Raw data normalization

    # r_data.plot_data(r_data.data[0:300])  # Plot normalized data
    # -------------------------------------------------------

    # Test of spike_detection class
    s = spikes.spike_detection(r_data.timeseries)
    filtered = s.filtering(4.0, win)  # Filter the normalized data
    sp0 = s.peaks(filtered, kind='aggregate')  # Peaks detection

    # Define the proper size of events positions- this must be done by the user
    positions = sp0[sp0 <= r_data.data_len/2.]

    # s.plot_filtered_data(s.data, filtered, 4.)  # Plot the data
    # s.plot_peaks(s.data, sp0)
    # -------------------------------------------------------

    # Test of events class
    evts = events.build_events(r_data.data, positions, win, before=14,
                               after=30)
    evtsE = evts.mk_events()    # Make spike events
    noise = evts.mk_noise()     # Make noise events

    # evts.plot_mad_median(evtsE)  # Plot mad and median of the events
    # evts.plot_events(evtsE)      # Plot events
    # -------------------------------------------------------

    # Test PCA and KMeans clustering
    CSize = 10
    c = clusters.pca_clustering(r_data.timeseries[0], positions, win, thr=8,
                                before=14, after=30)

    # print c.pca_variance(10)   # Print the variance of the PCs

    # c.plot_mean_pca()         # Plot the mean +- PC
    # c.plot_pca_projections()  # Plot the projections of the PCs on the data

    kmeans_clusters = c.KMeans(CSize)    # K-means clustering

    # gmm_clusters = c.GMM(10, 'diag')    # GMM clustering

    # tree = clusters.bagged_clustering(10, 100, 30)   # Bagged clustering

    # c.plot_clusters(gmm_clusters)   # Plot the clusters
    # -------------------------------------------------------

    # Test alignement of the spike events
    goodEvts = c.goodEvts
    align = alignment.align_events(r_data.timeseries[0], positions, goodEvts,
                                   kmeans_clusters, CSize, win)

    evtsE_noj = [align.mk_aligned_events(align.gcpos[i])
                 for i in range(CSize)]

    centers = {"Cluster " + str(i): align.mk_center_dictionary(evtsE_noj[i][1])
               for i in range(CSize)}

    round0 = [align.classify_and_align_evt(align.positions[i], centers)
              for i in range(len(align.positions))]

    print len([x[1] for x in round0 if x[0] == '?'])
    plt.show()
