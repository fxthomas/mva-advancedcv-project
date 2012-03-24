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

#########################
# Some useful functions #
#########################

def drawkp(im, kpl, rmin=5000):
  """
    Draws keypoint list `kpl` on image `im`, with minimum response `rmin` (default > 5000).
  """
  def x(kp):
    x,_ = kp.pt
    return x
  def y(kp):
    _,y = kp.pt
    return y

  arx = array ([x(kp) for kp in kpl if kp.response > rmin])
  ary = array ([y(kp) for kp in kpl if kp.response > rmin])
  hold(True)
  imshow (im)
  scatter (arx, ary)
  show()

def comparekp (left, right, kp1, kp2):
  """
  Compares keypoints by displaying them side by side.
  """
  subplot (121)
  arx = array ([kp1.pt[0]])
  ary = array ([kp1.pt[1]])
  hold(True)
  imshow(left)
  scatter (arx, ary)

  subplot (122)
  arx = array ([kp2.pt[0]])
  ary = array ([kp2.pt[1]])
  hold(True)
  imshow(right)
  scatter (arx, ary)

  show()

#################
# Rectification #
#################

def rectify (left, right, rmin=None):
  # Check image dimensions
  if left.shape != right.shape:
    raise ValueError ("left/right images must have the same dimensions")

  h,w,d = left.shape
  mask = ones((h,w), dtype=uint8)

  # Run a SURF detector on both images
  detector = cv2.SURF(0.)
  kp1, desc1 = detector.detect(uint8(left.mean(axis=2)).copy(), mask, False)
  kp2, desc2 = detector.detect(uint8(right.mean(axis=2)).copy(), mask, False)
  desc1 = desc1.reshape ((len(kp1), -1))
  desc2 = desc2.reshape ((len(kp2), -1))

  # Put descriptor responses in arrays
  resp1 = array([kp.response for kp in kp1])
  resp2 = array([kp.response for kp in kp2])

  #print ("Left : Found {0} descr. MIN/AVG/MAX:STD is {1} / {2} / {3} : {4}".format (len(kp1), resp1.min(), resp1.mean(), resp1.max(), resp1.std()))
  #print ("Right: Found {0} descr. MIN/AVG/MAX:STD is {1} / {2} / {3} : {4}".format (len(kp2), resp2.min(), resp2.mean(), resp2.max(), resp2.std()))
  #drawkp (left, kp1)

  # We want to keep only descriptors which are slightly (by std-dev) better than average
  rmin1 = rmin
  rmin2 = rmin
  if rmin is None:
    rmin1 = resp1.mean()
    rmin2 = resp2.mean()

  iok1 = find(resp1 > rmin1)
  iok2 = find(resp2 > rmin2)
  kp1 = array(kp1)[iok1]
  kp2 = array(kp2)[iok2]
  desc1 = desc1[iok1, :].copy()
  desc2 = desc2[iok2, :].copy()

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
  rv, h1, h2 = cv2.stereoRectifyUncalibrated (kp1a.transpose().reshape ((-1)), kp2a.transpose().reshape((-1)), ff, (h, w))

  left_r = left.copy()
  right_r = cv2.warpPerspective (right, np.linalg.inv(np.mat(h1)) * np.mat(h2), (w,h))
  return left_r, right_r

#########################
# Run from command line #
#########################

if __name__ == '__main__':
  # Read images and rectify them
  b1 = cv2.imread (sys.argv[1])
  b2 = cv2.imread (sys.argv[2])
  b1r, b2r = rectify (b1, b2)
