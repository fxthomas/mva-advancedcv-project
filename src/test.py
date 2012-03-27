#!/usr/bin/python
# coding=utf-8

# Base Python File (test.py)
# Created: Mon Mar 26 22:13:22 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

from pylab import *
import meanshift
import simpletree

# Load images
print ("Loading images...")
#left = imread ("data/t1.bmp")[::-1,:,:3]
#right = imread ("data/t2.bmp")[::-1,:,:3]
left = imread ("data/b1.jpg")[::-1,:,:]
right = imread ("data/b2.jpg")[::-1,:,:]

# Compute mean-shift segmentation
print ("Computing segmentation...")
segl = meanshift.segment (left)
segr = meanshift.segment (right)

# Show segmentation
subplot (221)
title ("Segmented left image")
imshow (segl)

subplot (222)
title ("Segmented right image")
imshow (segr)

# Compute disparity
print ("Computing disparity map...")
disp = simpletree.disparity (segl, segr)

# Show disparity map
print ("Displaying disparity map...")
subplot (223)
title ("Disparity map")
imshow (disp, cmap=cm.gray)

show()
