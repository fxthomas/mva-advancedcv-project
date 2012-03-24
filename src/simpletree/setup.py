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

dp = Extension ('dp',
                        sources=['dp_module.c', 'dp_methods.c'],
                        include_dirs = [np.get_include()],
                        extra_compile_args=["-O0"],
                        extra_link_args=["-O0"])

setup (name = "Simple tree DP",
       version = "0.1",
       description = "Simple tree DP algorithm for disparity map computation",
       ext_modules = [dp]
)
