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

LAMBDA = 1.

def disparitymap (left, right, hvcoeff=LAMBDA):
  ######################
  # Vertical tree pass #
  ######################

  print (" --> Vertical tree")

  # Horizontal pass
  F,m = simpletree.dp (im1[:,:,0], im2[:,:,0], axis=0, nd=20)
  B = simpletree.dp (im1[:,:,0], im2[:,:,0], energy=m, axis=0)
  C = F + B - m

  # Vertical pass
  Fc = simpletree.dp (im1[:,:,0], im2[:,:,0], energy=C, axis=1)
  Bc = simpletree.dp (im1[:,:,0], im2[:,:,0], energy=C, axis=1, backward=True)
  V = Fc + Bc - C

  ###################################
  # Compute subsequent coefficients #
  ###################################

  print (" --> Coefficients")
  Vc = m + hvcoeff*(V - V.min(axis=2).reshape((V.shape[0], V.shape[1], 1)))

  ########################
  # Horizontal tree pass #
  ########################

  print (" --> Horizontal tree")

  # Horizontal pass
  F = simpletree.dp (im1[:,:,0], im2[:,:,0], energy=Vc, axis=1)
  B = simpletree.dp (im1[:,:,0], im2[:,:,0], energy=Vc, axis=1, backward=True)
  C = F + B - Vc

  # Vertical pass
  Fc = simpletree.dp (im1[:,:,0], im2[:,:,0], energy=C, axis=0)
  Bc = simpletree.dp (im1[:,:,0], im2[:,:,0], energy=C, axis=0, backward=True)
  H = Fc + Bc - C

  return H.argmin(axis=2)

im1 = imread (sys.argv[1])
im2 = imread (sys.argv[2])
