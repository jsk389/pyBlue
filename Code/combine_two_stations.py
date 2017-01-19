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
from itertools import groupby

def get_bison(sfile, a, chunk_size=-1):
    """ Read in BiSON data from .h5 file """
    # TODO: chunksize needs testing
    if chunk_size > 0:
        data = pd.read_hdf(sfile, chunksize=chunk_size)
    else:
        data = pd.read_hdf(sfile)
    return data

def rebin(f, smoo):
    """ Rebin data over smoo bins """
    if smoo < 1.5:
        return f
    smoo = int(smoo)
    m = len(f) // smoo
    ff = np.median(f[:m*smoo].reshape((m,smoo)), axis=1)
    return ff

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

def find_overlaps(data1, data2):
    """ Find the where there are overlaps given two datasets.
        Returns indices where there are and are not overlaps
    """
    # Find overlapping regions
    overlap = np.where((data1[:,0] == data2[:,0]) &
                       (data1[:,1] != 0) & (data2[:,1] != 0))[0]
    # Get index array for station
    ia = np.indices(np.shape(data1[:,0]))
    # Work out where there are no overlaps
    no_overlap = np.setxor1d(ia, overlap)
    return overlap, no_overlap

def recover_overlaps(new_data, data1, data2, overlap):
    """ Recover overlap data, combine such that low frequency noise
        is minimised
    """
    # Combine overlapping regions
    overlap_starts = []
    overlap_ends = []
    nad = data1[:,1]
    sud = data2[:,1]
    for k, g in groupby(enumerate(overlap), lambda (i,y):i-y):
        x = map(itemgetter(1), g)
        overlap_starts.append(x[0])
        overlap_ends.append(x[-1])
        # Calculate fom for each station
        varn = calc_noise(nad[x])
        varx = calc_noise(sud[x])
        # Combine data
        combo = varn/(varn + varx) * nad[x] + varx/(varn + varx) * sud[x]
        new_data[x] = combo
        #print(varn/(varn + varx), varx/(varn + varx))
        #plt.plot(nad[x], 'b')
        #plt.plot(sud[x], 'g')
        #plt.plot(combo, 'r')
        #plt.show()
    return new_data, overlap_starts, overlap_ends

def add_in_no_overlaps(new_data, data1, data2, no_overlap):
    """ Add in regions of data where there are no overlaps
    """
    # Cut down data
    d = data1[no_overlap, 1]
    a = new_data[no_overlap]
    # Find where not equal to zero i.e. where taking data
    idxn = np.where(d != 0)
    # Add the data in
    a[idxn] = d[idxn]
    new_data[no_overlap] = a
    # Combine rest of data
    d = data2[no_overlap, 1]
    a = new_data[no_overlap]
    # Find where not equal to zero i.e. where taking data
    idxn = np.where(d != 0)
    # Add the data in
    a[idxn] = d[idxn]
    new_data[no_overlap] = a
    return new_data

def stitch_data(new_data, data1, data2, overlap, stitch_type='start'):
    """ Stitch together the data
        stitch_type keyword dictates whether stitching at the beginning of an
        overlap or at the end
    """
    t = data1[:,0]

    if stitch_type == 'start':
        # Stitch overlap starts
        for i in range(len(overlap)):
            # See which station is taking data before overlap
            if data1[overlap[i]-1,1] != 0:
                # Stitch data
                begin = overlap[i]-5
                finish = overlap[i]+5
                # Combine over 10 bins around start of overlap
                alpha = (t[1]-t[0])*5.0
                sigma =  1. / (1 + np.exp(-(t[begin:finish] - t[overlap[i]]) / alpha))

                new_data[begin:finish] = (1.0 - sigma) * data1[begin:finish, 1] \
                                                       + sigma * new_data[begin:finish]
            else:
                # Stitch data
                begin = overlap[i]-5
                finish = overlap[i]+5
                # Combine over 10 bins around start of overlap
                alpha = (t[1]-t[0])*5.0
                sigma =  1. / (1 + np.exp(-(t[begin:finish] - t[overlap[i]]) / alpha))

                new_data[begin:finish] = (1.0 - sigma) * data2[begin:finish, 1] + sigma * new_data[begin:finish]
    elif stitch_type == 'end':
        # Stitch overlap ends
        for i in range(len(overlap)):
            # See which station is taking data before overlap
            if data1[overlap[i]+1,1] != 0:
                # Stitch data
                begin = overlap[i]-5
                finish = overlap[i]+5
                # Combine over 10 bins around start of overlap
                alpha = (t[1]-t[0])*5.0
                sigma =  1. / (1 + np.exp(-(t[begin:finish] - t[overlap[i]]) / alpha))

                new_data[begin:finish] = (1.0 - sigma) * new_data[begin:finish] + sigma * data1[begin:finish, 1]

            else:
                # Stitch data
                begin = overlap[i]-5
                finish = overlap[i]+5
                # Combine over 10 bins around start of overlap
                alpha = (t[1]-t[0])*5.0
                sigma =  1. / (1 + np.exp(-(t[begin:finish] - t[overlap[i]]) / alpha))

                new_data[begin:finish] = (1.0 - sigma) * new_data[begin:finish] + sigma * data2[begin:finish, 1]

    return new_data

def run_concatenation(data1, data2, plot=False):
    """ Run concatenation procedure for 2 stations
    """
    # Find overlaps
    overlap, no_overlap = find_overlaps(data1, data2)

    # Create new data array
    new_data = np.zeros(len(data1))
    # Insert overlaps into new dataset
    new_data, overlap_starts, overlap_ends = recover_overlaps(new_data, data1, data2, overlap)
    #plt.plot(nat, nad, 'b')
    #plt.plot(sut, sud, 'g')
    #plt.plot(nat, new_data, 'r')
    #plt.show()
    # Combine data
    # Want indices data where not overlapping with any other station and then
    # set new data at those indices equal to value of data

    new_data = add_in_no_overlaps(new_data, data1, data2, no_overlap)
    new_data = stitch_data(new_data, data1, data2, overlap_starts, stitch_type='start')
    new_data = stitch_data(new_data, data1, data2, overlap_ends, stitch_type='end')
    print("STATION 1 FILL: ", float(len(data1[data1[:,1] != 0])) / float(len(data1)))
    print("STATION 2 FILL: ", float(len(data2[data2[:,1] != 0])) / float(len(data2)))

    print("NEW FILL: ", float(len(new_data[new_data != 0])) / float(len(new_data)))
    # Compare against naive case
    combo = np.c_[deepcopy(data1[:,1]), deepcopy(data2[:,1])]
    window = np.c_[deepcopy(data2[:,1]), deepcopy(data2[:,1])]
    window[:,0][window[:,0] != 0] = 1
    window[:,1][window[:,1] != 0] = 1
    combo = np.sum(combo, axis=1)/np.sum(window, axis=1)
    combo[~np.isfinite(combo)] = 0

    # Make sure there are no nans or infs
    new_data[~np.isfinite(new_data)] = 0
    if plot:
        # Compute FFTs
        fb, pb = FFT.fft_out.fft(40.0, len(combo), combo,
                                          'data', 'one')
        f, p = FFT.fft_out.fft(40.0, len(new_data), new_data,
                                          'data', 'one')
        plt.plot(f*1e6, pb-p)
        plt.title('Differences')
        plt.show()

        plt.title('')
        plt.plot(fb*1e6, pb, 'k')
        plt.plot(f*1e6, p, 'r')
        plt.show()

if __name__ == "__main__":
    print("Combine two station timeseries ...")

    # Individual sites
    directory = '/home/jsk389/Dropbox/Python/BiSON/SolarFLAG/PCA/Analysis/'
    fnames = ['ca_1yr_pca_ts.h5', 'cb_1yr_pca_ts.h5', 'iz_1yr_pca_ts.h5', \
              'la_1yr_pca_ts.h5', 'mo_1yr_pca_ts.h5', 'na_1yr_pca_ts.h5', \
              'su_1yr_pca_ts.h5']
    # Create labels from file names
    labels = []
    for i in fnames:
        labels.append(i.split('_')[0])

    station1 = 0
    station2 = 2
    # Read in data
    na = get_bison(str(directory)+str(fnames[int(station1)]), str(station1))
    na = np.c_[na['Time'].as_matrix(), na['Velocity'].as_matrix()]
    su = get_bison(str(directory)+str(fnames[int(station2)]), str(station2))
    su = np.c_[su['Time'].as_matrix(), su['Velocity'].as_matrix()]

    # Run procedure
    run_concatenation(na, su, plot=False)
