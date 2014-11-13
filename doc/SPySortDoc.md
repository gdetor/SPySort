# SPySort


## What is SPySort?

SPySort is a Python package for performing spike sorting. The underlying
implemented method is based on the spike sorting method proposed by Christophe
Pouzat et al (*Using noise signature to optimize spike-sorting and to assess 
neuronal classification quality*, Journal of Neuroscience Methods, 122: 43-57),
and it takes advantage of Numpy, Scipy, Pandas, Matplotlib and Scikit learn. 

## SPySort modules

**import_data.py:** This module offers a class of functions for reading, normalizing,
subsetting data, selecting specific recording channels, a summary function that
prints to the stdout all the important statistical numbers related to the raw data
and an interactive plot function. This module can be replaced at user's will.

**spikes.py:** This module includes all the necessary methods in order to filter
the raw data and to detect the spike events. In addition, two plot functions
are included. One for plotting the spike events and another one which plots the
spike events over the raw data and the threshold values as well.

**events.py:** In this module the user can find methods for extracting the
spike events from the raw normalized data. There are two main methods that make
the events and the noise. Moreover, two plot functions are included in order to
facilitate the visualization of the events and some statistical measures over 
the created events.

**clusters.py:** PCA based clustering is included in this module. A Principal
Component Analysis is applied on the events in order to reduce the dimension 
of the problem. In addition, in this package, three clustering methods have
been implemented. The basic clustering method is the k-means algorithm. The
second is a Gaussian Mixture model and finally a Bagged clustering algorithm 
Plot functions for the PCAs and the clustered events are also provided.

**alignment.py:** Because the spike events are not aligned the most of the 
times a brute-force algorithm of spike events alignment has been included in 
this module. Therefore, after the clustering of the spike events the user can 
refine the spike events and improve the clustering by using the methods provided
by this module.

## How to use SPySort?

SPySort can be used in a stand-alone software or in an interactive
environment, such as IPython.

> **Tip:** For use in IPython, check the corresponding notebook in the /doc 
        folder. It is strongly recommended that the user will read carefully 
        the extensive article provided by Christophe Pouzat:
        [SPySort Article](http://xtof.perso.math.cnrs.fr/pdf/PouzatDetorakis2014.pdf)




