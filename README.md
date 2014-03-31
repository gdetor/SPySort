SPySort
-------

SPySort is a Python package for spikes sorting. The package has the ability to
read raw neurophysiological data of extracellular recordings in binary format. 
Although, the user is not forced to use the modules of SPySort in order to read
and preprocess their raw data. The spike sorting method that is mainly
implemented is given in [1]. 

The modules of SPySort are:
i. data.py : This module offers a class of functions for reading,
normalizing, subsetting data, selecting specific recording channels, a summary
functions that prints to the stdout all the important statistical numbers
related to the raw data and an interactive plot function. 

ii. spikes.py : This module includes all the necessary methods in order
to filter the raw data and to detect the spike events. In addition, two plot
functions are included. One for plotting the spike events and another one which
plots the spike events over the raw data and the threshold values as well. 

iii. events.py : In this module the user can find methods for extracting the 
spike events from the raw normalized data. There are two main methods that make 
the events and the noise. Moreover, two plot functions are included in order to
facilitate the visualization of the events and some statistical measures over
the created events. 

iv. clusters.py : PCA based clustering is included in this module. A Principal
Component Analysis is applied on the events in order to reduce the dimension of
the problem and then a plethora of clustering methods are implemented. The
basic clustering method is the k-means algorithm. In addition, a Gaussian
Mixture model and a Bagged clustering algorithm have been implemented, as well.
Plot functions for the PCAs and the clustered events are also provided. 

v. alignment.py : Because the spike events are not aligned the most of the times
a brute-force algorithm of spike events alignment has been included in this
module. Therefore, after the clustering of the spike events the user can refine
the spike events and improve the clustering by using the methods provided by
this module. 


References:

[1] Pouzat, C., Mazor, O. & Laurent, G., "Using noise signature to optimize 
spike-sorting and to assess neuronal classification quality", Journal of 
neuroscience methods, 122, pp.43–57, 2002.
