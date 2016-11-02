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

def calc_noise(data):
    """ Calculate noise using low frequency figure-of-merit """
    # If there are enough points then run FFT, else just take variance
    if len(data) > 125:
        f, p = FFT.fft_out.fft(40.0, len(data), data,
                                  'data', 'one')
        start = 0.0
        end  = 500.0e-6 / (f[1]-f[0])
        noise = np.sum(p[start:end+1]*(f[1]-f[0]))
    else:
        noise = np.var(data)
    return 1.0 / noise

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

def compute_all_combinations(n_stations):
    """ Compute all combinations of stations for overlap search
    """
    combos = []
    for i in range(2, n_stations+1):
        combos.append(list(combinations(range(n_stations), i)))

    return list(chain(*combos))

def find_overlaps(data, n_stations):
    """ Find the where there are overlaps given n_station datasets.
        Returns indices where there are and are not overlaps
    """
    # Number of stations for combining
    combinations = compute_all_combinations(n_stations)

    overlaps = {}
    for i in range(len(combinations)):
        tmp_dat = data[:,combinations[i]]
        cond = []
        overlaps[len(combinations[i])] = np.where(data[:,])
    # Find overlapping regions
    overlap = np.where((data1[:,0] == data2[:,0]) &
                       (data1[:,1] != 0) & (data2[:,1] != 0))[0]
    # Get index array for station
    ia = np.indices(np.shape(data1[:,0]))
    # Work out where there are no overlaps
    no_overlap = np.setxor1d(ia, overlap)
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
    station1 = 0
    station2 = 2
    station3 = 3
    stations = [station1, station2, station3]
    fnames = [fnames[station1], fnames[station2], fnames[station3]]
    # Read in data
    data = read_data(directory, fnames, stations)

    # Plot data
    #plot_data(data)
    #plt.show()

    # Run procedure
    run_concatenation(data, plot=False)
