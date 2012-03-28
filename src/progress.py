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
  def __init__ (self, message, ntick=0):
    self.value = 0
    self.message = message
    self._started = False
    self._increment = 100./ntick

  def start (self):
    if self._started:
      self.end()

    stdout.write (self.message + " [    %]")
    stdout.flush()
    self._started = True
    self.value = 0

  def tick (self, value=None):
    if not self._started:
      self.start()

    if value is not None:
      self.value = value
    else:
      self.value = self.value + self._increment

    stdout.write ("\b\b\b\b\b\b{0:3.0f} %]".format (self.value))
    stdout.flush()

  def end (self):
    if self._started:
      stdout.write ("\b\b\b\b\b\bDone] \n")
      stdout.flush()
      self._started = False

  def __enter__ (self):
    self.start()
    return self

  def __exit__ (self, exc_type, exc_value, traceback):
    self.end()
