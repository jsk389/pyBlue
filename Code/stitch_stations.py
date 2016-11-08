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
