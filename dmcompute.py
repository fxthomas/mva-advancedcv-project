#!/usr/bin/python
# coding=utf-8

# Base Python File (dmcompute.py)
# Created: Fri Mar 16 14:27:45 2012
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
import sys
import simpletree

# Read images
im1 = imread (sys.argv[1])[::-1,:,:]
im2 = imread (sys.argv[2])[::-1,:,:]

# Forward pass
F,m = simpletree.dp (im1[:,:,0], im2[:,:,0], nd=20, backward=False, return_point_energy=True)

# Backward pass
B = simpletree.dp (im1[:,:,0], im2[:,:,0], nd=20, backward=True)

# Compute optimal costs
C = F + B - m

# Display computed disparity map (with only per-scanline optimization)
imshow (C.argmin(axis=2), cmap=cm.gray)
show()
