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

LAMBDA = 0.025
P1 = 20.
P2f = 30.
P3 = 4
T = 30

im1 = imread (sys.argv[1]).mean (axis=2)
im2 = imread (sys.argv[2]).mean (axis=2)

######################
# Vertical tree pass #
######################

print (" --> Vertical tree")

# Horizontal pass
m = simpletree.data_energy (im1, im2, nd=20, axis=1)

F = simpletree.dp (im1, im2, energy=m, axis=0, P1=P1, P2f=P2f, P3=P3, T=T)
B = simpletree.dp (im1, im2, energy=m, axis=0, backward=True, P1=P1, P2f=P2f, P3=P3, T=T)
C = F + B - m

# Vertical pass
Fc = simpletree.dp (im1, im2, energy=C, axis=1, P1=P1, P2f=P2f, P3=P3, T=T)
Bc = simpletree.dp (im1, im2, energy=C, axis=1, backward=True, P1=P1, P2f=P2f, P3=P3, T=T)
V = Fc + Bc - C

###################################
# Compute subsequent coefficients #
###################################

print (" --> Coefficients")
Vc = m + LAMBDA*(V - V.min(axis=2).reshape((V.shape[0], V.shape[1], 1)))

########################
# Horizontal tree pass #
########################

print (" --> Horizontal tree")

# Horizontal pass
F = simpletree.dp (im1, im2, energy=Vc, axis=1, P1=P1, P2f=P2f, P3=P3, T=T)
B = simpletree.dp (im1, im2, energy=Vc, axis=1, backward=True, P1=P1, P2f=P2f, P3=P3, T=T)
C = F + B - Vc

# Vertical pass
Fc = simpletree.dp (im1, im2, energy=C, axis=0, P1=P1, P2f=P2f, P3=P3, T=T)
Bc = simpletree.dp (im1, im2, energy=C, axis=0, backward=True, P1=P1, P2f=P2f, P3=P3, T=T)
H = Fc + Bc - C

Dmap = H.argmin(axis=2)

imshow (Dmap, cmap=cm.gray)
show()
