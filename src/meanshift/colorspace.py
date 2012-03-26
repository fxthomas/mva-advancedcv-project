#!/usr/bin/python
# coding=utf-8

# Base Python File (colorspace.py)
# Created: Mon Mar 26 19:40:59 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

import numpy as np

def rgb2luv (image):
  if image.ndim != 3:
    raise ValueError("image must have 3 color channels")

  iluv = np.float_(image.copy())
  if iluv.max() > 1.:
    iluv = iluv / 255.

  XYZ = np.array ([[.4125, .3576, .1804],
                   [.2125, .7154, .0721],
                   [.0193, .1192, .9502]])
  Yn = 1.0
  Lt = .008856
  Up = 0.19784977571475
  Vp = 0.46834507665248

  h,w,_ = iluv.shape
  iluv = np.rollaxis(iluv, 2, 0).reshape((3, -1))

  xyz = np.dot (XYZ, iluv).transpose().reshape ((h, w, 3))
  x = xyz[:,:,0]
  y = xyz[:,:,1]
  z = xyz[:,:,2]

  l0 = y / Yn
  l = l0

  l[l0 > Lt] = 116 * (l0[l0 > Lt])**(1./3.) - 16.
  l[l0 <= Lt] = 903.3 * (l0[l0 <= Lt])

  c = x + 15*y + 3*z
  u = 4. * np.ones((h,w))
  v = (9./15.) * np.ones((h,w))
  u[c!=0] = 4 * x[c!=0]/c[c!=0]
  v[c!=0] = 9 * y[c!=0]/c[c!=0]
  
  u = 13 * l * (u - Up)
  v = 13 * l * (v - Vp)

  return np.rollaxis(np.array ([l, u, v]), 0, 3)

def luv2rgb (image):
  if image.ndim != 3:
    raise ValueError("image must have 3 color channels")

  irgb = np.float_(image.copy())

  RGB = np.array ([[ 3.2405, -1.5371, -0.4985],
                   [-0.9693,  1.8760,  0.0416],
                   [ 0.0556, -0.2040,  1.0573]]);

  Up = 0.19784977571475;
  Vp = 0.46834507665248;
  Yn = 1.00000;

  l = irgb[:,:,0]
  y = Yn * l / 903.3
  y[l > .8] = (l[l > .8] + 16.) / 116.
  y[l > .8] = Yn * y[l > .8]**3

  u = Up + irgb[:,:,1] / (13. * l)
  v = Vp + irgb[:,:,2] / (13. * l)

  x = 9. * u * y / (4*v)
  z = (12. - 3.*u - 20.*v) * y / (4*v)

  h = image.shape[0]
  w = image.shape[1]
  d = image.shape[2]

  x = x.reshape ((h,w,1))
  y = y.reshape ((h,w,1))
  z = z.reshape ((h,w,1))

  rgb = np.rollaxis(np.concatenate ((x, y, z), axis=2), 2, 0).reshape ((3, -1))
  rgb = np.dot (RGB, rgb)
  rgb = rgb.transpose().reshape(image.shape)

  zr = np.where(l < .1)
  rgb[zr[0], zr[1], :] = 0
  rgb[rgb < 0] = 0
  rgb[rgb > 1] = 1

  return rgb
