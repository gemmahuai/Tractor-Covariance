import sys
import astrometry
import numpy as np
import timeit
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