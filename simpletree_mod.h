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
static PyObject *dp_pass(PyObject *self, PyObject *args, PyObject *kwdict);

/**
 * Exported method list
 */
static PyMethodDef Methods[] = {
  {"dp", (PyCFunction)dp_pass, METH_KEYWORDS | METH_VARARGS, "Computes the disparity map of the 2 rectified stereo images in argument."},
  {NULL, NULL, 0, NULL}
};
