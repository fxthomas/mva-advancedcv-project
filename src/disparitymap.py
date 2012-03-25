#!/usr/bin/python
# coding=utf-8

# Base Python File (disparitymap.py)
# Created: Sun Mar 25 16:36:35 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

import simpletree
import cv2
import sys
from pylab import *

# Load left-right images
left = cv2.imread (sys.argv[1])
right = cv2.imread (sys.argv[2])

# Compute disparity
disp = simpletree.disparity (left, right)

# Show disparity map
imshow (disp, cmap=cm.gray)
show()
