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

import fast_lnp
def PS_PS(freq, power, ulim, llim):

	# Take power evenly spaced in period and compute average power spectrum according to period spacing

	bw = freq[1]-freq[0]

	f_ulim = ulim/bw
	f_llim = llim/bw

	period = np.ones(len(freq[f_llim:f_ulim]))/freq[f_llim:f_ulim]
	powers = power[f_llim:f_ulim]

	# window
	px, py, nout, jmax, prob = fast_lnp.fasper(period, powers, 10.0, 1.0)

	return px, py


if __name__=="__main__":
    import scipy.ndimage

    print("Stitching!")
    fnames = glob.glob("/home/jsk389/Dropbox/Python/BiSON/SolarFLAG/PCA/Analysis/new_combined_ts/*.h5")
    #fnames = glob.glob("/home/jsk389/Dropbox/Python/BiSON/SolarFLAG/PCA/Analysis/new_ts/su*.h5")

    fnames = np.sort(fnames)
    # First cycle outside loop
    data1 = get_bison(fnames[0])
    data2 = get_bison(fnames[1])
    t = data1[:,0]
    t = np.append(t, data2[:,0])
    new_dat = data1[:,1]
    new_dat = np.append(new_dat, data2[:,1])


    for i in range(2, len(fnames)):
        print("Stitching: ", i)
        data = get_bison(fnames[i])
        t = np.append(t, data[:,0])
        new_dat = np.append(new_dat, data[:,1])
    #df = pd.DataFrame(data=np.c_[t, new_dat], columns=['Time', 'Velocity'])
    #df.to_hdf('/home/jsk389/Dropbox/Python/BiSON/SolarFLAG/PCA/Analysis/combined_pca_ts.h5', 'a', mode='w', format='fixed')


    f, p = FFT.fft_out.fft(40.0, len(new_dat), new_dat,
                                      'data', 'one')
    # PSPS
    #smoop = scipy.ndimage.filters.uniform_filter1d(p, int(20.0e-6/(f[1]-f[0])))
    #px, py = PS_PS(f, p/smoop, 140.0e-6, 60.0e-6)
    #plt.plot((1.0/px) / 60.0, py, 'k')
    #for i in range(1, 5):
    #    plt.axvline((34.49/np.sqrt(2.0))/float(i), color='r', linestyle='--')
    #for i in range(1, 5):
    #    plt.axvline((34.49/2.0)/float(i), color='g', linestyle='--')
    #plt.ylabel(r'PSPS')
    #plt.xlim(0, 50)
    #plt.xlabel(r'Period (minutes)')
    #plt.show()


    plt.title('')
    plt.plot(f*1e6, p, 'k')
    plt.xlim(1, f.max()*1e6)
    plt.ylim(1e-4, 1e6)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
