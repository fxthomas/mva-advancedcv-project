#!/usr/bin/python
# coding=utf-8

# Base Python File (progress.py)
# Created: Wed Mar 28 16:00:24 2012
# Version: 1.0
#
# This Python script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.
# If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
# 
# (ɔ) François-Xavier Thomas <fx.thomas@gmail.com>

from sys import stdout

class ProgressMeter:
  def __init__ (self, message):
    self.value = 0.
    self.message = message
    self._started = False

  def start (self):
    stdout.write (self.message + " [    %]")
    stdout.flush()
    self._started = True

  def tick (self, value):
    if not self._started:
      self.start()

    stdout.write ("\b\b\b\b\b\b{0:3.0f} %]".format (value))
    stdout.flush()

  def end (self):
    if self._started:
      stdout.write ("\b\b\b\b\b\bDone] \n")
      stdout.flush()

  def __enter__ (self):
    self.start()
    return self

  def __exit__ (self, exc_type, exc_value, traceback):
    self.end()
