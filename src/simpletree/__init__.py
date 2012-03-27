#!/usr/bin/python
# coding=utf-8

# Base Python File (__init__.py)
# Created: Fri Mar 16 14:27:45 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

import dp

def disparity (left, right, nd=20, LAMBDA=0.025, P1=20, P2f=30, P3=4, T=30):
  """
  Computes the disparity map of two rectified stereo images `left` and `right`.

  `nd` is the maximum value of the disparity labels (-nd <= d < nd)
  `LAMBDA` is the horizontal/vertical DP balancing term
  `P1`, `P2f`, `P3`, `T` are the smoothness model paramters
  """

  if left.ndim == 3:
    left = left.mean (axis=2)
  if right.ndim == 3:
    right = right.mean (axis=2)
  if left.max() <= 1.1:
    left = left * 255.
  if right.max() <= 1.1:
    right = right * 255.

  ######################
  # Vertical tree pass #
  ######################

  # Horizontal pass
  m = dp.data_energy (left, right, nd=nd)

  F = dp.dp (left, right, energy=m, axis=0)
  B = dp.dp (left, right, energy=m, axis=0, backward=True)
  C = F + B - m

  # Vertical pass
  Fc = dp.dp (left, right, energy=C, axis=1)
  Bc = dp.dp (left, right, energy=C, axis=1, backward=True)
  V = Fc + Bc - C

  ###################################
  # Compute subsequent coefficients #
  ###################################

  Vc = m + LAMBDA*(V - V.min(axis=2).reshape((V.shape[0], V.shape[1], 1)))

  ########################
  # Horizontal tree pass #
  ########################

  # Horizontal pass
  F = dp.dp (left, right, energy=Vc, axis=1)
  B = dp.dp (left, right, energy=Vc, axis=1, backward=True)
  C = F + B - Vc

  # Vertical pass
  Fc = dp.dp (left, right, energy=C, axis=0)
  Bc = dp.dp (left, right, energy=C, axis=0, backward=True)
  H = Fc + Bc - C

  return H.argmin(axis=2)
