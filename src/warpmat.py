#!/usr/bin/python
# coding=utf-8

# Base Python File (warpmat.py)
# Created: Wed Mar 28 15:22:41 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

# For displaying stuff on the console
from __future__ import with_statement
from progress import ProgressMeter
from sys import stdout

# Array & Image manipulation libraries
from pylab import *
from scipy.ndimage import generate_binary_structure, morphology
import matplotlib

# Planar regression
from sklearn import linear_model

# Meanshift segmentation module
import meanshift

# SimpleTree disparity map algorithm module
import simpletree

np.seterr (invalid="ignore")

# Load images
left, right = None, None
with ProgressMeter ("Loading images", 2) as p:
  #left = imread ("data/t1.bmp")[::-1,:,:3]
  #left = imread ("data/m1r.jpg")[::-1,:,:]
  left = imread ("data/b1.jpg")[::-1,:,:]
  p.tick()

  #right = imread ("data/t2.bmp")[::-1,:,:3]
  #right = imread ("data/m2r.jpg")[::-1,:,:]
  right = imread ("data/b2.jpg")[::-1,:,:]
  p.tick()

# Compute mean-shift segmentation
segl,segr,labels = None,None,None
with ProgressMeter ("Computing segmentation", 2) as p:
  segl,labels = meanshift.segment (left, return_labels=True)
  p.tick()

  segr = meanshift.segment (right)
  p.tick()

# Compute disparity
disp = None
with ProgressMeter ("Computing disparity map", 1) as p:
  disp = simpletree.disparity (segl, segr)
  p.tick()

# Prepare coordinates array
xp = arange (0, 1, 1./labels.shape[1])
yp = arange (0, 1, 1./labels.shape[0])
xxp,yyp = meshgrid (xp, yp)
xxp = xxp.reshape ((-1,1))
yyp = yyp.reshape ((-1,1))
data_x = concatenate ((xxp,yyp), axis=1)
data_y = labels.transpose().reshape ((-1,))
models = zeros ((labels.max()+1, 3))

dmax = data_y.max()
with ProgressMeter ("Computing segment coefficients", dmax+1) as p:
  for fl in range (dmax+1):
      p.tick ()

      mdl = linear_model.SGDRegressor (loss='squared_loss', penalty='l1')
      mdl.fit (data_x[data_y==fl,:], data_y[data_y==fl])
      models[fl,0] = mdl.coef_[0]
      models[fl,1] = mdl.coef_[1]
      models[fl,2] = mdl.intercept_

# Generate left/right buffers
left_buffer = zeros ((labels.shape[0], labels.shape[1], 40, 4)) # We have ND=20 in the SimpleTree disparity algorithm
right_buffer = zeros ((labels.shape[0], labels.shape[1], 40, 4))

# Generate initial data for left buffer
#st = generate_binary_structure (2, 1)
with ProgressMeter ("Generating buffers", 40) as p:
  for d in range(40):
    p.tick()

    left_buffer[:,:,d,0][disp == d] = segl[:,:,0][disp == d]
    left_buffer[:,:,d,1][disp == d] = segl[:,:,1][disp == d]
    left_buffer[:,:,d,2][disp == d] = segl[:,:,2][disp == d]
    left_buffer[:,:,d,3][disp == d] = 1.

    #_leftd = morphology.binary_dilation (left_buffer[:,:,d,3], structure=st, iterations=5)
    #_lefte = morphology.binary_erosion (left_buffer[:,:,d,3], structure=st, iterations=5)
    #left_buffer[:,:,d,3][_leftd & ~_lefte] = 0.5


virt_right_image = zeros (left.shape)
coeff = zeros ((left.shape[0], left.shape[1]))
with ProgressMeter ("Composing right image", 40) as p:
  for di in range(40):
    p.tick()
    d = di - 20

    tmp = np.roll (left_buffer[:,:,di,:], d, axis=1)
    virt_right_image = virt_right_image + tmp[:,:,:3]*tmp[:,:,3].reshape((tmp.shape[0],tmp.shape[1],1))
    coeff = coeff + tmp[:,:,3]
virtr = virt_right_image / coeff.reshape((coeff.shape[0],coeff.shape[1],1))
virtr[isnan(virtr)] = 1.

subplot (221)
title ("Segmented left image")
imshow (labels, cmap=matplotlib.colors.ListedColormap (np.random.rand (256, 3)))

subplot (222)
title ("Disparity map")
imshow (disp, cmap=cm.gray)

subplot (223)
title ("Reconstructed right image")
imshow (virtr)

subplot (224)
title ("Right view and reconstruction overlay")
imshow (right)
imshow (virt_right_image / coeff.reshape((coeff.shape[0],coeff.shape[1],1)), alpha=0.5)

show()
