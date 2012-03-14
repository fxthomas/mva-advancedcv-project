#!/usr/bin/python
# coding=utf-8

# Base Python File (rectify.py)
# Created: Wed Mar 14 20:17:10 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

from numpy import *
from scipy import *
from pylab import *
import cv2
import sys
import numpy as np

# Some useful functions
def x(kp):
  x,_ = kp.pt
  return x
def y(kp):
  _,y = kp.pt
  return y
def drawkp(im, kpl, rmin=5000):
  """
    Draws keypoint list `kpl` on image `im`, with minimum response `rmin` (default > 5000).
  """
  arx = array ([x(kp) for kp in kpl if kp.response > rmin])
  ary = array ([y(kp) for kp in kpl if kp.response > rmin])
  hold(True)
  imshow (im)
  scatter (arx, ary)
  show()

# Read images and initialize stuff
b1 = imread ("b1.jpg")[::-1,:]
b2 = imread ("b2.jpg")[::-1,:]
if b2.shape != b1.shape:
  print ("Error: b1.shape != b2.shape!...")
  sys.exit(1)
h,w,d = b1.shape
mask = 255*ones((h,w), dtype=uint8)

# Run a SURF detector on both images
detector = cv2.SURF()
kp1, desc1 = detector.detect(b1[:,:,0].copy(), mask, False)
kp2, desc2 = detector.detect(b2[:,:,0].copy(), mask, False)
desc1 = desc1.reshape ((-1, 128))
desc2 = desc2.reshape ((-1, 128))
#drawkp (b1, kp1)

# Match descriptors, see http://www.maths.lth.se/matematiklth/personal/solem/book.html for more info
matchidx = -1 * ones ((len(desc1)), 'int')
desc2t = desc2.transpose()
dist_ratio = 0.6
for i in range(len(desc1)):
  dotprods = dot (desc1[i,:], desc2t) * 0.9999
  acdp = arccos (dotprods)
  index = argsort (acdp)

  if acdp[index[0]] < dist_ratio * acdp[index[1]]:
    matchidx[i] = index[0]

# Only keep matched descriptors
kp1 = [kp1[i] for i in range(len(matchidx)) if matchidx[i] >= 0]
kp2 = [kp2[matchidx[i]] for i in range(len(matchidx)) if matchidx[i] >= 0]
kp1a = array ([kp.pt for kp in kp1])
kp2a = array ([kp.pt for kp in kp2])

# Rectify images
ff = cv2.findFundamentalMat (kp1a, kp2a)[0]
rv, h1, h2 = cv2.stereoRectifyUncalibrated (kp1a.reshape ((-1)), kp2a.reshape((-1)), ff, (640, 480))
subplot (121); imshow (cv2.warpPerspective (b1, h1, (640, 480)))
subplot (122); imshow (cv2.warpPerspective (b2, h2, (640, 480)))
show()
