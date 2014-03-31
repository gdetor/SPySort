import numpy as np
import pandas as pd
from events import events
from numpy.linalg import svd
import matplotlib.pylab as plt
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import pdist
from functions import mad, good_evts_fct
from pandas.tools.plotting import scatter_matrix
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


class pca_clustering(events):
    """ Clustering methods and dimension-reduction techniques """
    def __init__(self, data, positions, win, thr=8, before=14, after=30):
        """ Performs the cleaning of the events and a singular value
        decomposition in order to obtain the principal components of the
        data.

        **Parameters**

        IFiles : string (list)
            Contains the full paths of the input files

        freq : double
            The sampling frequency of the original data in order to create
            the proper timeseries.

        win : double (array)
            The filtering window (can be a boxcar, winner, etc)

        thr : double
            The threshold value used during filtering

        before : int
            The number of sampling point to keep before the peak

        after : int
            The number of sampling point to keep after the peak
        """
        events.__init__(self, data, positions, win, before, after)
        # Convert the list of events to a numpy array
        self.evts = np.asarray(self.mk_events())
        # Convert the list of noise events to a numpy array
        self.noise = np.asarray(self.mk_noise())
        # Compute the clean events
        # self.goodEvts = good_evts_fct(self.evts, thr)
        self.goodEvts = self.sieve(good_evts_fct, self.evts, thr)
        # Compute the covariance matrix
        varcovmat = np.cov(self.evts[self.goodEvts, :].T)
        # Perform a singular value decomposition
        self.U, self.S, self.V = svd(varcovmat)

    def plot_mean_pca(self):
        """ Plots the mean of the data plus-minus the principal components """
        evt_idx = range(self.evts.shape[1])
        evts_good_mean = np.mean(self.evts[self.goodEvts, :], 0)
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.plot(evt_idx, evts_good_mean, 'k',
                     evt_idx, evts_good_mean + 5 * self.U[:, i], 'r',
                     evt_idx, evts_good_mean - 5 * self.U[:, i], 'b')
            plt.title('PC' + str(i) + ': ' + str(round(self.S[i]/sum(self.S) *
                      100)) + '%')

    def pca_variance(self, n_pca):
        """ Returns the variance of the principal components.

        **Parameters**

        n_pca : int
            Number of principal components to be taken into account
        """
        noiseVar = sum(np.diag(np.cov(self.noise.T)))
        evtsVar = sum(self.S)
        return [(i, sum(self.S[:i]) + noiseVar - evtsVar) for i in
                range(n_pca)]

    def plot_pca_projections(self, pca_components=(0, 4)):
        """ Plots the principal components projected on the data.

        **Parameters**

        pca_components : int (tuple)
            The number of the principal components to be projected
        """
        tmp = np.dot(self.evts[self.goodEvts, :],
                     self.U[:, pca_components[0]:pca_components[1]])
        df = pd.DataFrame(tmp)
        scatter_matrix(df, alpha=.2, s=4, c='k', figsize=(8, 8),
                       diagonal='kde', marker=".")

    def KMeans(self, n_clusters, init='k-means++', n_init=100, max_iter=100,
               n_pca=(0, 3)):
        """ It computes the k-means clustering over the dimension-reducted
            data.

        **Parameters**

        n_clusters : int
            The number of the clusters

        init : string
            Method for initialization (see scikit-learn K-Means for more
            information)

        n_init : int
            Number of time the k-means algorithm will be run with different
            centroid seeds

        max_iter : int
            Maximum number of iterations of the k-means algorithm for a
            single run

        n_pca : int (tuple)
            Chooses which PCs are used
        """
        km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                    max_iter=max_iter)
        km.fit(np.dot(self.evts[self.goodEvts, :],
                      self.U[:, n_pca[0]:n_pca[1]]))
        c = km.fit_predict(np.dot(self.evts[self.goodEvts, :],
                                  self.U[:, n_pca[0]:n_pca[1]]))

        c_med = list([(i, np.apply_along_axis(np.median, 0,
                       self.evts[self.goodEvts, :][c == i, :])) for i in
                      range(10) if sum(c == i) > 0])
        c_size = list([np.sum(np.abs(x[1])) for x in c_med])
        new_order = list(reversed(np.argsort(c_size)))
        new_order_reverse = sorted(range(len(new_order)),
                                   key=new_order.__getitem__)
        return [new_order_reverse[i] for i in c]

    def GMM(self, n_comp, cov_type, n_iter=100, n_init=100, init_params='wmc',
            n_pca=(0, 3)):
        """ It clusters the data points using a Gaussian Mixture Model.

        ** Parameters **

        n_comp : int
            Number of mixture components

        cov_type : string
            Covarianve parameters to use

        n_iter : int
            Number of EM iterations to perform

        n_init : int
            Number of initializations to perform

        init_params : string
            Controls which parameters are updated in the training process.

        n_pca : int (tuple)
            Controls which PCs are used
        """
        gmm = GMM(n_components=n_comp, covariance_type=cov_type, n_iter=n_iter,
                  n_init=n_init, init_params=init_params)

        gmm.fit(np.dot(self.evts[self.goodEvts, :],
                       self.U[:, n_pca[0]:n_pca[1]]))

        c = gmm.predict(np.dot(self.evts[self.goodEvts, :],
                               self.U[:, n_pca[0]:n_pca[1]]))

        c_med = list([(i, np.apply_along_axis(np.median, 0,
                       self.evts[self.goodEvts, :][c == i, :])) for i in
                      range(10) if sum(c == i) > 0])
        c_size = list([np.sum(np.abs(x[1])) for x in c_med])
        new_order = list(reversed(np.argsort(c_size)))
        new_order_reverse = sorted(range(len(new_order)),
                                   key=new_order.__getitem__)
        return [new_order_reverse[i] for i in c]

    # TODO: To finish the bagged clustering routine
    def bagged_clustering(self, n_bootstraps, n_samples, n_iter,
                          show_dendro=False, n_pca=(0, 3)):
        """ Performs a bagged clustering (using hierarchical clustering and
            k-means) on the events data.

        ** Parameters **

        n_bootstraps : int
            Number of bootstraped samples to create

        n_samples : int
            Number of samples each bootstraped set contains

        n_iter : int
            The maximum number of k-Means iterations

        show_dendro : boolean
            If it's true the method displays the dendrogram

        n_pca : int (tuple)
            The number of PCs which are used
        """

        B, N = n_bootstraps, n_samples
        data = np.dot(self.evts[self.goodEvts, :],
                      self.U[:, n_pca[0]:n_pca[1]])
        size_r, size_c = data.shape[0], data.shape[1]

        if n_samples > data.shape[0]:
            print 'Too many sample points'
            return -1

        # Construct B bootstrap training samples and run the base cluster
        # method - KMeans
        C = []
        for i in range(B):
            centroids, _ = kmeans(data[np.random.randint(0, size_r, (N,)), :],
                                  k_or_guess=N, iter=n_iter)
            C.extend(centroids)

        # Run a hierarchical clustering
        distMatrix = pdist(C, 'euclidean')
        D = linkage(distMatrix, method='single')

        # Create the dendrogram
        if show_dendro == 'True':
            dendrogram(D)

        # Cut the tree
        F = fcluster(D, 2, criterion='maxclust')
        return F

    def plot_event(self, x, n_plot=None, events_color='black', events_lw=0.1,
                   show_median=True, median_color='red', median_lw=0.5,
                   show_mad=True, mad_color='blue', mad_lw=0.5):
        """ Plots an event after clustering.

        **Parameters**

        x : double (list or array)
            Data to be plotted
        n_plot : int
            Number of events that will be plotted
        events_color : string
            Lines color
        events_lw : float
            Line width
        show_median : boolean
            If it's True the median will appear in the figure
        median_color : strin
            Median curve color
        median_lw : float
            Median curve width
        show_mad : boolean
            It it's true the mad will appear in the figure
        mad_color : string
            Mad curve color
        mad_lw : float
            Mad curve width
        """
        x = np.asarray(x)

        if n_plot is None:
            n_plot = x.shape[0]

        for i in range(n_plot):
            plt.plot(x[i, :], color=events_color, lw=events_lw)

        if show_median:
            MEDIAN = np.apply_along_axis(np.median, 0, x)
            plt.plot(MEDIAN, color=median_color, lw=median_lw)

        if show_mad:
            MAD = np.apply_along_axis(mad, 0, x)
        plt.plot(MAD, color=mad_color, lw=mad_lw)

        plt.axvspan(45, 89, fc='grey', alpha=.5, edgecolor='none')
        plt.axvspan(135, 179, fc='grey', alpha=.5, edgecolor='none')

    def plot_clusters(self, clusters):
        """ Plots events belong to five different clusters.

        **Parameters**

        clusters : int (array or list)
            The index of the cluster from which the events will be plotted
        """
        plt.subplot(511)
        self.plot_event(self.evts[self.goodEvts, :]
                        [np.array(clusters) == 0, :])
        plt.ylim([-15, 20])
        plt.subplot(512)
        self.plot_event(self.evts[self.goodEvts, :]
                        [np.array(clusters) == 1, :])
        plt.ylim([-15, 20])
        plt.subplot(513)
        self.plot_event(self.evts[self.goodEvts, :]
                        [np.array(clusters) == 2, :])
        plt.ylim([-15, 20])
        plt.subplot(514)
        self.plot_event(self.evts[self.goodEvts, :]
                        [np.array(clusters) == 3, :])
        plt.ylim([-15, 20])
        plt.subplot(515)
        self.plot_event(self.evts[self.goodEvts, :]
                        [np.array(clusters) == 4, :])
        plt.ylim([-15, 20])
