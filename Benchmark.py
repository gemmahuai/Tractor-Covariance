import sys
import astrometry
import numpy as np
import timeit
import time
import seaborn as sns
from astropy.stats import sigma_clipped_stats
from astropy import wcs
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.coordinates import SkyCoord
from astropy.io import fits
import pickle
import math
import pdb
import random
import warnings
import pandas as pd
import os
import matplotlib.pyplot as plt
import importlib
import copy
from astropy.nddata import Cutout2D
import scipy.optimize as opt

import matplotlib as mpl
from scipy.stats import binned_statistic
import scipy.optimize as opt
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)
plt.rcParams['font.size'] = 12
plt.rcParams["figure.figsize"] = (6,4)

pi = math.pi

#Importing Simulator 
#import SPHEREx_SkySimulator as SPsky
import SPHEREx_ObsSimulator as SPobs
import SPHEREx_InstrumentSimulator as SPinst
from SPHEREx_Simulator_Tools import SPHEREx_Logger
from pkg_resources import resource_filename
from astropy.io import fits
import pandas as pd
from scipy.signal import convolve2d
from skimage.transform import downscale_local_mean
from astropy.stats import sigma_clipped_stats
from SPHEREx_Simulator_Tools import data_filename
 

from tractor_utils import *

import astropy.units as u

Logger = SPHEREx_Logger()

instrument = SPinst.Instrument(psf=data_filename('psf/simulated_PSF_database.fits'),
                               psf_downsample_by_array = {1:4, 2:4, 3:4, 4:2, 5:2, 6:2},
                               psf_trim_by_array = {1:32, 2:32, 3:32, 4:32, 5:32, 6:32},
                               noise_model = SPinst.white_noise,
                               dark_current_model = SPinst.poisson_dark_current,
                               lvf_model = SPinst.smile_lvf,
                               Logger=Logger)

which_calc = int(sys.argv[1])


### COVARIANCE CALCULATION ####

def Covariance_calc(umod, sigma, noise_map):
    """
    umodels = 3D array containing pixel weightings of each source
    sigma = sky std
    -----
    returns the fisher matrix, covariance matrix, and flux error
    -----
    `which_calc` = 0
    """
    D = umod.shape[0] # number of sources
    F = np.zeros(shape=(D+1,D+1)) # initialize fisher information matrix
    for i in range(D):
        F[i, D] = - np.sum(umod[i]*noise_map)  / sigma**2
        F[D, i] = F[i, D]
        for j in range(D):
            # i,j th entry of the fisher matrix
            F[i,j] = - np.sum(umod[i]*umod[j]) / sigma**2        
    # print('F = ', F)
    
    # check if F is invertible or not
    if np.linalg.det(F)==0.:
        print('F not invertible!')
        C = np.inf + np.zeros_like(F)
        var = -np.diag(C)
    else:
        C = np.linalg.inv(F)
        var = -np.diag(C) #variance
        
    ### calculate F1, F2, F2/F1
    
        
    # print('Covariance matrix = ', C)
    return(F, C, var)

def Covariance_calc_rank1Decomp(umod, sigma, noise_map):
    """
    Another way of calculating matrix inverse, using Rank 1 decomposition, 
    ouelined in the ML notes. 
    Hopefully faster. 
    `which_calc` = 1
    """
    D = umod.shape[0] # number of sources
    F = np.zeros(shape=(D+1,D+1)) # initialize fisher information matrix

    # symmetric fisher information matrix
    for i in range(D):
        F[i, D] = - np.sum(umod[i]*noise_map)  / sigma**2
        F[D, i] = F[i, D]
        for j in range(D):
            # i,j th entry of the fisher matrix
            F[i, j] = - np.sum(umod[i]*umod[j]) / sigma**2

    D_inv = 1/np.diag(F)[:-1]
    z = F[-1][:-1]
    u = D_inv * z
    u = np.append(u, -1)
    alpha = F[-1,-1]
    rho = 1 / (alpha - np.dot(z, D_inv*z))
    C = np.zeros_like(F) + rho * u * u
    np.fill_diagonal(C, np.append(D_inv, 0))
    var = -np.diag(C) #variance

    return(F, C, var)



### generate multiple psf's

def PSF_gen_multiple(N_psf, Coord, array_number, N_image, plot=False):
    """
    N_psf = number of psf's to generate
    Coord = an N_psf dimentional tuple - coordinates of these psf's (we place the first psf at the center)
    array_number = SPHEREx array number
    N_image = size of the image containing these sources
    -----
    returns N_psf * N_image * N_image array containing psf's;
    plots the oversampled and downscaled psf's in the same image
    """
    (x0, y0) = Coord[0] # the first psf coord
    downscale = 5
    psf_ds = np.zeros(shape=(N_psf, int(N_image/downscale), int(N_image/downscale)))
    #for rendering
    psf_im_plt = np.zeros(shape=(int(N_image), int(N_image)))
    psf_ds_plt = np.zeros(shape=(int(N_image/downscale), int(N_image/downscale)))
    
    for i in range(N_psf):
        (x_i, y_i) = Coord[i]
        xoff = x_i - x0
        yoff = y_i - y0
        psf_i = instrument.PSF.psf(x_i, y_i, array=array_number)
        im_i = np.zeros((N_image, N_image))
        s = int(psf_i.shape[0]/2.)
        im_i[(int(N_image/2)+1+yoff-s):(int(N_image/2)+1+yoff+s),
            (int(N_image/2)+xoff-s):(int(N_image/2)+xoff+s)] += psf_i.T
        psf_i_ds = (downscale_local_mean(im_i,
                                         (downscale, downscale))
                    * downscale**2)
        psf_ds[i] = psf_i_ds
        
        # for rendering
        psf_im_plt += im_i
        psf_ds_plt += psf_i_ds

    
    if plot is not False:
        plt.figure(figsize=(10,3.5))
        plt.subplot(1,2,1)
        plt.imshow(psf_im_plt)
#         plt.plot(50,50, '*', color='white')
#         plt.xlim(40,60)
#         plt.ylim(40,60)
        plt.colorbar()
        plt.title('Oversampled PSF')
        plt.subplot(1,2,2)
        plt.imshow(psf_ds_plt)
        plt.colorbar()
        plt.title('Downsampled PSF')
        plt.show()
        
    return(psf_ds)



if __name__ == '__main__':
    
    ### how the performace scales with the number of nearby sources
    num_runs = 1000
    N_sources = 20
    N_timeavg = 1000
    im_sigma = 1.
    # put the main source at the center
    coord = [(100,100)]
    runtime = []

    for i in range(N_sources):
        coord_x, coord_y = np.random.randint(80, 120, size=2)
        coord.append((coord_x, coord_y))
    
    umod = PSF_gen_multiple(N_psf=N_sources, 
                            Coord=coord, 
                            array_number=4,
                            N_image=100, 
                            plot=False)
    
    noise = np.random.normal(scale = im_sigma, size=(umod.shape[1], umod.shape[2])) * 0.05

    ## Start timing!
    
    time_start = time.time()
    
    if which_calc == 0:
        for j in range(num_runs):
            Covariance_calc(umod, im_sigma, noise)

        time_end = time.time()
        runtime_avg = (time_end - time_start) * 1e6 / num_runs; # us
    elif which_calc == 1:
        for j in range(num_runs):
            Covariance_calc_rank1Decomp(umod, im_sigma, noise)

        time_end = time.time()
        runtime_avg = (time_end - time_start) * 1e6 / num_runs; # us


    print(f'Averaged execution time over {num_runs}: {runtime_avg:.3f} us. ')

