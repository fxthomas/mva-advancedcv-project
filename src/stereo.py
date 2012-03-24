#!/usr/bin/python
# coding=utf-8

# Base Python File (stereo.py)
# Created: Sat Mar 24 17:47:41 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

from pylab import *

def show_stereo (left, right):
  subplot (121)
  imshow (left)
  subplot (122)
  imshow (right)
  show()
