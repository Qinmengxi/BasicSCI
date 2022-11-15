#!/usr/bin/env python
# coding: utf-8

# ## GAP-TV for Video Compressive Sensing
# ### GAP-TV
# > X. Yuan, "Generalized alternating projection based total variation minimization for compressive sensing," in *IEEE International Conference on Image Processing (ICIP)*, 2016, pp. 2539-2543.
# ### Code credit
# [Xin Yuan](https://www.bell-labs.com/usr/x.yuan "Dr. Xin Yuan, Bell Labs"), [Bell Labs](https://www.bell-labs.com/), xyuan@bell-labs.com, created Aug 7, 2018.  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, Tsinghua University"), [Tsinghua University](http://www.tsinghua.edu.cn/publish/thu2018en/index.html), y-liu16@mails.tsinghua.edu.cn, updated Jan 20, 2019.

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import time
import math
import h5py
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean

from pnp_sci import admmdenoise_cacti

from utils import (A_, At_)


# In[2]:


# [0] environment configuration
datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
resultsdir = './results' # results

alldatname= []

for dir_name in os.listdir(datasetdir):
    if(dir_name.endswith('.mat')):
        alldatname.append(dir_name)

for datname in alldatname:

    matfile = datasetdir + '/' + datname

    nframe = -1
    print(datname.split('.')[0])

    from scipy.io.matlab.mio import _open_file
    from scipy.io.matlab.miobase import get_matfile_version

    # [1] load data
    if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
        file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
        meas = np.float32(file['meas'])
        mask = np.float32(file['masks'])
        orig = np.float32(file['orig'])
    else: # MATLAB .mat v7.3
        file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
        meas = np.float32(file['meas']).transpose()
        mask = np.float32(file['masks']).transpose()
        orig = np.float32(file['orig']).transpose()

    print(meas.shape, mask.shape, orig.shape)

    iframe = 0
    if nframe < 0:
        nframe = meas.shape[2]
    MAXB = 255.

    # common parameters and pre-calculation for PnP
    # define forward model and its transpose
    A  = lambda x :  A_(x, mask) # forward model function handle
    At = lambda y : At_(y, mask) # transpose of forward model

    mask_sum = np.sum(mask, axis=2)
    mask_sum[mask_sum==0] = 1


    # [2] GAP-TV
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    denoiser = 'tv' # total variation (TV)
    iter_max = 40 # maximum number of iterations
    tv_weight = 0.3 # TV denoising weight (larger for smoother but slower)
    tv_iter_max = 5 # TV denoising maximum number of iterations each

    vgaptv,orig_,tgaptv,psnr_gaptv,ssim_gaptv,psnrall_gaptv = admmdenoise_cacti(meas, mask, A, At,
                                             projmeth=projmeth, v0=None, orig=orig,
                                             iframe=iframe, nframe=nframe,
                                             MAXB=MAXB, maskdirection='plain',
                                             _lambda=_lambda, accelerate=accelerate,
                                             denoiser=denoiser, iter_max=iter_max, 
                                             tv_weight=tv_weight, 
                                             tv_iter_max=tv_iter_max)

    print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
       projmeth.upper(), denoiser.upper(), mean(psnr_gaptv), mean(ssim_gaptv), tgaptv))

    # [3] result demonstration of GAP-TV
    nmask = mask.shape[2]
    
    print(resultsdir + '/target')
    
    if not os.path.exists(resultsdir + '/target'):
        os.mkdir(resultsdir + '/target')
    if not os.path.exists(resultsdir + '/input'):
        os.mkdir(resultsdir + '/input')
    if not os.path.exists(resultsdir + '/target/' + datname.split('.')[0]):
        os.makedirs(resultsdir + '/target/' + datname.split('.')[0])
    if not os.path.exists(resultsdir + '/input/' + datname.split('.')[0]):
        os.makedirs(resultsdir + '/input/' + datname.split('.')[0])

    for i in range(orig_.shape[2]):
        cv2.imwrite(resultsdir + '/' + 'target/' + datname.split('.')[0] + '/' + str(i).zfill(5) + '.png', orig_[:,:,i])
        cv2.imwrite(resultsdir + '/' + 'input/' + datname.split('.')[0] + '/' + str(i).zfill(5) + '.png', vgaptv[:,:,i]*255)
