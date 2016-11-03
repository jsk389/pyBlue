#!/usr/bin/env python3
from __future__ import division

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import os
import pyfits
import jdcal
import FFT

import pandas as pd

from operator import itemgetter
from itertools import groupby, permutations, chain

def get_bison(sfile, a, chunk_size=-1):
    """ Read in BiSON data from .h5 file """
    # TODO: chunksize needs testing
    if chunk_size > 0:
        data = pd.read_hdf(sfile, chunksize=chunk_size)
    else:
        data = pd.read_hdf(sfile)
    return data

def read_data(directory, fname, station):
    """ Fetch BiSON data and return time series
    """
    if len(fname) > 1:
        # If want to read in more than one station at a time
        data = get_bison(str(directory)+str(fname[0]), str(station[0]))
        tmp = np.c_[data['Time'].as_matrix(), data['Velocity'].as_matrix()]
        data = np.zeros([len(tmp), len(fname)+1])
        data[:,:2] = tmp
        for i in range(1, len(fname)):
            dat = get_bison(str(directory)+str(fname[i]), str(station[i]))
            tmp = dat['Velocity'].as_matrix()
            data[:,1+i] = tmp

        return data
    else:
        data = get_bison(str(directory)+str(fname), str(station))
        data = np.c_[data['Time'].as_matrix(), data['Velocity'].as_matrix()]
        return data

def plot_data(data):
    plt.plot(data[:,0], data[:,1])
    plt.plot(data[:,0], data[:,2])
    plt.plot(data[:,0], data[:,3])

def combinations(iterable, r):
    """ Function taken from python itertools docs
        combinations(range(4), 3) --> 012 013 023 123
    """
    pool = tuple(iterable)
    n = len(pool)
    for indices in permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)

def create_selectors(active_stations, n_stations):
    """ Generate a list of stations of length n_stations which is 1
        where stations is chosen and 0 otherwise
        Input: tuple containing number of each active stations (0, to n_stations-1)
    """
    selectors = np.zeros(n_stations, dtype=int)
    for i in active_stations:
        selectors[i] = 1
    return selectors.tolist()

def compress(data, selectors):
    """ Choose data given selectors, taken from python itertools docs.
    This enables us to select the data we want given the selectors
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    """
    return (d for d, s in zip(data, selectors) if s)

def compute_all_combinations(n_stations):
    """ Compute all combinations of stations for overlap search
    """
    combos = []
    for i in range(2, n_stations+1):
        combos.append(list(combinations(range(n_stations), i)))

    return list(chain(*combos))

def where_overlaps(data):
    """ Given a data set, return where there are overlaps
    """
    x = np.zeros(len(data))
    # Create boolean array for each column
    for i in range(1, np.shape(data)[1]):
        x += 1*((data[:,(i-1)] != 0) & (data[:, i] != 0))
    # Normalise to account for shorter station overlaps than those wanted
    x /= (float(np.shape(data)[1] - 1))
    # The above effectively works out to summing the window function and
    # normalising by the number of stations so that were it equals one
    # is equivalent to the overlap we are interested in

    # Overlaps are now where x == 1
    overlaps = np.where(x == 1)[0]

    # Get index array for station
    ia = np.indices(np.shape(data[:,0]))
    # Work out where there are no overlaps
    no_overlap = np.setxor1d(ia, overlaps)

    return np.where(x == 1)[0], no_overlap

def calc_noise(data):
    """ Calculate noise using low frequency figure-of-merit """
    # If there are enough points then run FFT, else just take variance
    noise = np.zeros(np.shape(data)[1])
    for i in range(len(noise)):
        if len(data) > 125:
            f, p = FFT.fft_out.fft(40.0, len(data[:,i]), data[:,i],
                                    'data', 'one')
            start = 0.0
            end  = 500.0e-6 / (f[1]-f[0])
            noise[i] = np.sum(p[start:end+1]*(f[1]-f[0]))
        else:
            noise[i] = np.var(data[:,i])
    return 1.0 / noise

def recover_overlaps(new_data, data, overlap):
    """ Recover overlapping data such that low frequency noise is minimised
    """
    # Combine overlapping regions
    overlap_starts = []
    overlap_ends = []
    # Define overlapping regions
    for k, g in groupby(enumerate(overlap), lambda (i,y): i-y):
        x = map(itemgetter(1), g)
        overlap_starts.append(x[0])
        overlap_ends.append(x[-1])
        # Calculate the fom for given stations
        fom = calc_noise(data[x, :])
        # Combine datasets
        combo = np.array([fom[i]/np.sum(fom) * data[x,i]
                          for i in range(np.shape(data)[1])]).T.sum(1)
        print(np.shape(combo))
        #for i in range(np.shape(data[x,:])[1]):
    #        plt.plot(data[x,i])
#        plt.plot(combo, 'r', linestyle='--')
        #plt.show()
        new_data[x] = combo
    return new_data, overlap_starts, overlap_ends

def find_overlaps(data, n_stations):
    """ Find the where there are overlaps given n_station datasets.
        Returns indices where there are and are not overlaps
    """
    # Number of stations for combining
    combinations = compute_all_combinations(n_stations)
    # Create new data array
    new_data = np.zeros(len(data))
    # Loop over all combinations
    for i in range(len(combinations)):
        print(combinations[i])
        # Create selectors
        selectors = create_selectors(combinations[i], n_stations)
        # Use selectors to choose data from correct stations - WORKS!
        x = compress(data[:,1:].T, selectors)
        x = np.array(list(x)).T
        # Add time array back in, just in case it is needed
        tmp_data = np.c_[data[:,0], x]
        print(np.shape(x))
        overlaps, no_overlaps = where_overlaps(tmp_data[:,1:])
        # Add in data from overlaps
        new_data, _, _ = recover_overlaps(new_data, tmp_data[:,1:], overlaps)
    plt.plot(new_data)
    plt.show()
    sys.exit()


    # TODO: set up a dictionary containing number of stations overlapping and
    # where and when they start and finish

    sys.exit()


    overlaps = {}
    for i in range(len(combinations)):
        tmp_dat = data[:,combinations[i]]
        cond = []
        overlaps[len(combinations[i])] = np.where(data[:,])
    # Find overlapping regions
    overlap = np.where((data1[:,0] == data2[:,0]) &
                       (data1[:,1] != 0) & (data2[:,1] != 0))[0]

    return overlap, no_overlap

def run_concatenation(data, plot=False):
    """ Run concatenation procedure
    """

    # Work out number of stations - subtract one due to time axis
    n_stations = np.shape(data)[1] - 1

    # Work out overlaps
    find_overlaps(data, n_stations)

if __name__=="__main__":

    print("Combine three station timeseries ...")

    # Individual sites
    directory = '/home/jsk389/Dropbox/Python/BiSON/SolarFLAG/PCA/Analysis/'
    fnames = ['ca_1yr_pca_ts.h5', 'cb_1yr_pca_ts.h5', 'iz_1yr_pca_ts.h5', \
              'la_1yr_pca_ts.h5', 'mo_1yr_pca_ts.h5', 'na_1yr_pca_ts.h5', \
              'su_1yr_pca_ts.h5']
    # Create labels from file names
    labels = []
    for i in fnames:
        labels.append(i.split('_')[0])
    # Select stations to combine
    station1 = 2
    station2 = 3
    station3 = 4

    stations = [station1, station2, station3]
    stations = [0,1,2,3,4,5]
    fnames = [fnames[i] for i in stations]
    print(fnames)
    #fnames = [fnames[station1], fnames[station2], fnames[station3]]
    # Read in data
    data = read_data(directory, fnames, stations)

    # Plot data
    #plot_data(data)
    #plt.show()

    # Run procedure
    run_concatenation(data, plot=False)
