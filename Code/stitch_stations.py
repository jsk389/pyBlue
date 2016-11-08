#!/usr/bin/env python3
from __future__ import division

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import os
import pyfits
import jdcal
import FFT
import glob
import pandas as pd

from operator import itemgetter
from itertools import groupby, permutations, chain

def get_bison(sfile, chunk_size=-1):
    """ Read in BiSON data from .h5 file """
    # TODO: chunksize needs testing
    if chunk_size > 0:
        data = pd.read_hdf(sfile, chunksize=chunk_size)
    else:
        data = pd.read_hdf(sfile)
        data = np.c_[data['Time'].as_matrix(), data['Velocity'].as_matrix()]

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


def stitch(data1, data2, t_stitch):
    """ Stitch together the data where we assume data1 is the data preceeding
        the stitching point (t_stitch) and data2 is the data coming afterwards.
        Initially we shall assuming we are stitching 1 year time series together
        and therefore (annoyingly) hard code this!
    """
    print("Stitching")
    # Number of points in year
    npts = 2160.0 * 365.0
    # Create new data arrays
    new_data = np.concatenate([data1[:npts], data2[-npts:]])
    # Extract time arrays
    t1 = data1[:,0]
    t2 = data2[:,0]
    # We only want section of the time arrays that overlap
    idx = np.where(t1 == t2)
    # Create new shortened time array
    t = t1[idx]
    # Stitching parameters
    # Stitch over a region of +/- 40 bins (around 4 alpha)
    alpha = (t[1]-t[0])*10.0
    sigma = 1. / (1 + np.exp(-(t - t_stitch)/alpha))
    # Stitch together
    new_data[idx] = (1.0 - sigma) * data1[idx, 1] + \
                    sigma * data2[idx, 1]
    return new_data



if __name__=="__main__":

    print("Stitching!")
    fnames = glob.glob("/home/jsk389/Dropbox/Python/BiSON/SolarFLAG/PCA/Analysis/new_combined_ts/*.h5")
    fnames = np.sort(fnames)

    # First cycle outside loop
    data1 = get_bison(fnames[0])
    data2 = get_bison(fnames[1])
    print(data1)
    print(data1[:,0])
    idx = np.where(data1[:,0] == data2[:,0])[0]
    print(idx)
    t_stitch = data1[idx[len(idx//2)], 0]

    print(np.shape(data1))
    print(t_stitch)
    sys.exit()

    for i in range(1, len(fnames)):
        data1 = get_bison(fnames[i-1])
        data2 = get_bison(fnames[i])
        print(np.shape(data1))
        sys.exit()
        #new_data = stitch(data1, data2,)
