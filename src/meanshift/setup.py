#!/usr/bin/python
# coding=utf-8

# Base Python File (setup.py)
# Created: Sat Mar 17 15:41:11 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

from distutils.core import setup, Extension
import numpy as np

edison = Extension ('edison',
                        sources = ['edison_wrapper.cpp', 'edison/segm/ms.cpp', 'edison/segm/msImageProcessor.cpp', 'edison/segm/msSysPrompt.cpp', 'edison/segm/RAList.cpp', 'edison/segm/rlist.cpp', 'edison/edge/BgEdge.cpp', 'edison/edge/BgImage.cpp', 'edison/edge/BgGlobalFc.cpp', 'edison/edge/BgEdgeList.cpp', 'edison/edge/BgEdgeDetect.cpp'],
                        include_dirs = [np.get_include()],
                        extra_compile_args=["-O0"],
                        extra_link_args=["-O0"])

setup (name = "EDISON",
       version = "0.1",
       description = "Simple wrapper for EDISON for Python",
       ext_modules = [edison]
)
