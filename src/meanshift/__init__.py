#!/usr/bin/python
# coding=utf-8

# Base Python File (__init__.py)
# Created: Mon Mar 26 21:59:08 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

import edison
import colorspace
import numpy as np

def segment(image):
  iluv = np.float32 (colorspace.rgb2luv (image))
  fim, labels, modes = edison.meanshift (iluv)
  return colorspace.luv2rgb(fim)
