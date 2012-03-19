/**
 * Base C Header (simpletree.h)
 * Created: Sat Mar 17 15:49:59 2012
 *
 * This C Header was developped by François-Xavier Thomas.
 * You are free to copy, adapt or modify it.
 * If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
 * 
 * @author François-Xavier Thomas <fx.thomas@gmail.com>
 * @version 1.0
 */

#include <Python.h>
#include <numpy/arrayobject.h>

/**
 * Module initialization
 */
PyMODINIT_FUNC initsimpletree (void);

/**
 * Declare module methods
 */
static PyObject *imagedp(PyObject *self, PyObject *args, PyObject *kwdict);
static PyObject *dp(PyObject *self, PyObject *args, PyObject *kwdict);

/**
 * Exported method list
 */
static PyMethodDef Methods[] = {
  {"imagedp", (PyCFunction)imagedp, METH_KEYWORDS | METH_VARARGS, "Computes the first DP passes for the 2 rectified stereo images in argument."},
  {"dp", (PyCFunction)dp, METH_KEYWORDS | METH_VARARGS, "Computes a DP pass with pre-computed pixel energies."},
  {NULL, NULL, 0, NULL}
};
