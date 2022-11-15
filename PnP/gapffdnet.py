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

    # In[3]:
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
    # nframe = 1
    # nframe = meas.shape[2]
    if nframe < 0:
        nframe = meas.shape[2]
    MAXB = 255.

    # common parameters and pre-calculation for PnP
    # define forward model and its transpose
    A  = lambda x :  A_(x, mask) # forward model function handle
    At = lambda y : At_(y, mask) # transpose of forward model

    mask_sum = np.sum(mask, axis=2)
    mask_sum[mask_sum==0] = 1

    import torch
    from packages.ffdnet.models import FFDNet

    # [2] GAP-FFDNet (FFDNet-based frame-wise video denoising)
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    denoiser = 'ffdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
    iter_max = [10, 10, 10, 10] # maximum number of iterations
    # sigma    = [12/255, 6/255] # pre-set noise standard deviation
    # iter_max = [10,10] # maximum number of iterations
    useGPU = True # use GPU

    # pre-load the model for FFDNet image denoising
    in_ch = 1
    model_fn = 'packages/ffdnet/models/net_gray.pth'
    # Absolute path to model file
    # model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)

    # Create model
    net = FFDNet(num_input_channels=in_ch)
    # Load saved weights
    if useGPU:
       state_dict = torch.load(model_fn)
       device_ids = [0]
       model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
    else:
       state_dict = torch.load(model_fn, map_location='cpu')
       # CPU mode: remove the DataParallel wrapper
       state_dict = remove_dataparallel_wrapper(state_dict)
       model = net
    model.load_state_dict(state_dict)
    model.eval() # evaluation mode

    vgapffdnet,orig_,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                             projmeth=projmeth, v0=None, orig=orig,
                                             iframe=iframe, nframe=nframe,
                                             MAXB=MAXB, maskdirection='plain',
                                             _lambda=_lambda, accelerate=accelerate,
                                             denoiser=denoiser, model=model, 
                                             iter_max=iter_max, sigma=sigma)

    print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
       projmeth.upper(), denoiser.upper(), mean(psnr_gapffdnet), mean(ssim_gapffdnet), tgapffdnet))


    # [3] result demonstration of FFDNet
    nmask = mask.shape[2]
    
    print(resultsdir + '/target')
    
    if not os.path.exists(resultsdir + '/target'):
        print(os.path.exists(resultsdir + '/target'))
        os.mkdir(resultsdir + '/target')
        print(os.path.exists(resultsdir + '/target'))
    if not os.path.exists(resultsdir + '/input'):
        os.mkdir(resultsdir + '/input')
    if not os.path.exists(resultsdir + '/target/' + datname.split('.')[0]):
        os.makedirs(resultsdir + '/target/' + datname.split('.')[0])
    if not os.path.exists(resultsdir + '/input/' + datname.split('.')[0]):
        os.makedirs(resultsdir + '/input/' + datname.split('.')[0])

    for i in range(orig_.shape[2]):
        cv2.imwrite(resultsdir + '/' + 'target/' + datname.split('.')[0] + '/' + str(i).zfill(5) + '.png', orig_[:,:,i])
        cv2.imwrite(resultsdir + '/' + 'input/' + datname.split('.')[0] + '/' + str(i).zfill(5) + '.png', vgapffdnet[:,:,i]*255)

