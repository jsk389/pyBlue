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


def get_bison(sfile, a):
    data = pd.HDFStore(sfile)
    return data[str(a)]


def rebin(f, smoo):
    if smoo < 1.5:
        return f
    smoo = int(smoo)
    m = len(f) // smoo
    ff = np.median(f[:m*smoo].reshape((m,smoo)), axis=1)
    return ff

def calc_noise(data):
    if len(data) > 40:
        f, p = FFT.fft_out.fft(40.0, len(nad[x]), nad[x],
                                  'data', 'one')
        start = 0.0
        end  = 200.0e-6 / (f[1]-f[0])
        print(start, end)
        noise = np.sum(p[start:end+1]*(f[1]-f[0]))
    else:
        noise = np.var(data)
    return 1.0 / noise

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

    na = get_bison(str(directory)+str(fnames[-2]), '5')
    na = np.c_[na['Time'].as_matrix(), na['Velocity'].as_matrix()]
    su = get_bison(str(directory)+str(fnames[-1]), '6')
    su = np.c_[su['Time'].as_matrix(), su['Velocity'].as_matrix()]

    nat = na[:,0]
    sut = su[:,0]
    nad = na[:,1]
    sud = su[:,1]

    # Find overlapping regions
    overlap = np.where((na[:,0] == su[:,0]) & (na[:,1] != 0) & (su[:,1] != 0))[0]
    # Get index array for station
    ia = np.indices(np.shape(na[:,0]))
    # Work out where there are no overlaps
    no_overlap = np.setxor1d(ia, overlap)
    #no_overlap = list(set(ia.tolist())-set(overlap))
    # New data array
    new_data = np.zeros(len(na))
    # Combine overlapping regions
    overlap_starts = []
    overlap_ends = []
    for k, g in groupby(enumerate(overlap), lambda (i,y):i-y):
        x = map(itemgetter(1), g)
        overlap_starts.append(x[0])
        overlap_ends.append(x[-1])
        # Calculate fom for each station
        varn = calc_noise(nad[x])
        varx = calc_noise(sud[x])

        # Combine data
        combo = varn/(varn + varx) * nad[x] + varx/(varn + varx) * sud[x]
        #plt.plot(nat[x], nad[x], 'b')
        #plt.plot(sut[x], sud[x], 'g')
        #plt.plot(nat[x], combo, 'r')
        new_data[x] = combo
        #plt.show()


    # Combine data
    # Want indices data where not overlapping with any other station and then
    # set new data at those indices equal to value of data

    # Cut down data
    d = na[no_overlap, 1]
    a = new_data[no_overlap]
    # Find where not equal to zero i.e. where taking data
    idxn = np.where(d != 0)
    # Add the data in
    a[idxn] = d[idxn]
    new_data[no_overlap] = a
    # Combine rest of data
    d = su[no_overlap, 1]
    a = new_data[no_overlap]
    # Find where not equal to zero i.e. where taking data
    idxn = np.where(d != 0)
    # Add the data in
    a[idxn] = d[idxn]
    new_data[no_overlap] = a

    # Stitch data together properly
    t = na[:,0]
    # Stitch overlap starts
    for i in range(len(overlap_starts)):
        # See which station is taking data before overlap
        if na[overlap_starts[i]-1,1] != 0:
            # Stitch data
            begin = overlap_starts[i]-5
            finish = overlap_ends[i]+5
            # Combine over 10 bins around start of overlap
            alpha = (t[1]-t[0])*5.0
            sigma =  1. / (1 + np.exp(-(t[begin:overlap_starts[i]+5] - t[overlap_starts[i]]) / alpha))

            new_data[begin:overlap_starts[i]+5] = (1.0 - sigma) * na[begin:overlap_starts[i]+5, 1] + sigma * new_data[begin:overlap_starts[i]+5]

        else:
            # Stitch data
            begin = overlap_starts[i]-5
            finish = overlap_ends[i]+5
            # Combine over 10 bins around start of overlap
            alpha = (t[1]-t[0])*5.0
            sigma =  1. / (1 + np.exp(-(t[begin:overlap_starts[i]+5] - t[overlap_starts[i]]) / alpha))

            new_data[begin:overlap_starts[i]+5] = (1.0 - sigma) * su[begin:overlap_starts[i]+5, 1] + sigma * new_data[begin:overlap_starts[i]+5]

    # Stitch overlap ends
    for i in range(len(overlap_ends)):
        # See which station is taking data before overlap
        if na[overlap_ends[i]+1,1] != 0:
            # Stitch data
            begin = overlap_ends[i]-5
            finish = overlap_ends[i]+5
            # Combine over 10 bins around start of overlap
            alpha = (t[1]-t[0])*5.0
            sigma =  1. / (1 + np.exp(-(t[begin:finish] - t[overlap_ends[i]]) / alpha))

            new_data[begin:finish] = (1.0 - sigma) * new_data[begin:finish] + sigma * na[begin:finish, 1]

        else:
            # Stitch data
            begin = overlap_ends[i]-5
            finish = overlap_ends[i]+5
            # Combine over 10 bins around start of overlap
            alpha = (t[1]-t[0])*5.0
            sigma =  1. / (1 + np.exp(-(t[begin:finish] - t[overlap_ends[i]]) / alpha))

            new_data[begin:finish] = (1.0 - sigma) * new_data[begin:finish] + sigma * su[begin:finish, 1]





    print("STATION 1 FILL: ", float(len(na[na[:,1] != 0])) / float(len(na)))
    print("STATION 2 FILL: ", float(len(su[su[:,1] != 0])) / float(len(su)))

    print("NEW FILL: ", float(len(new_data[new_data != 0])) / float(len(new_data)))
    combo = np.c_[deepcopy(na[:,1]), deepcopy(su[:,1])]
    window = np.c_[deepcopy(na[:,1]), deepcopy(su[:,1])]
    window[:,0][window[:,0] != 0] = 1
    window[:,1][window[:,1] != 0] = 1
    combo = np.sum(combo, axis=1)/np.sum(window, axis=1)
    combo[~np.isfinite(combo)] = 0
    plt.plot(combo)
    plt.show()
    new_data[~np.isfinite(new_data)] = 0
    fb, pb = FFT.fft_out.fft(40.0, len(combo), combo,
                                          'data', 'one')
    f, p = FFT.fft_out.fft(40.0, len(new_data), new_data,
                                          'data', 'one')
    plt.plot(f*1e6, pb-p)
    plt.show()

    plt.plot(fb*1e6, pb, 'k')
    plt.plot(f*1e6, p, 'r')
    plt.show()
