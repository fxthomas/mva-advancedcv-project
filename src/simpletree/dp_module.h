/**
 * Base C Header (dp_module.h)
 * Created: Sat Mar 17 15:49:59 2012
 *
 * This C Header was developped by François-Xavier Thomas.
 * You are free to copy, adapt or modify it.
 * If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
 * 
 * @author François-Xavier Thomas <fx.thomas@gmail.com>
 * @version 1.0
 */

// Multiple file includes for Numpy : http://www.gossamer-threads.com/lists/python/python/59597
#define PY_ARRAY_UNIQUE_SYMBOL __PyArraySimpleTree

#include <Python.h>
#include <numpy/arrayobject.h>

/**
 * Module initialization
 */
PyMODINIT_FUNC initdp (void);

/**
 * Declare module methods
 */
PyDoc_STRVAR (Py_dp_doc, "dp (left, right, [energy], [backward=False], [nd=10], [axis=1], [P1=20], [P2f=30], [P3=4], [T=30])\n\
\n\
  Computes a DP pass for the disparity map computation algorithm\n\
  described in \"Simple but effective tree structure for DP-based stero matching\"\n\
  by Michael Bleyer and Margrit Gelautz.\n\
  \n\
  `left` and `right` are the left/right rectified stero images\n\
  `energy` is a pre-computed by-pixel energy (optional)\n\
      If this argument is not present, the `dp` function will compute\n\
      a default energy function, and return it as a (F, m) tuple.\n\
  `backward` is True means this is a backward pass,\n\
      otherwise it defaults to forward\n\
  `nd` is the max absolute value of the disparity\n\
      (will be overwritten if energy is present)\n\
  `axis` is the tree structure to use for the algorithm\n\
      (scanlines are horizontal if axis=1, vertical otherwise)\n\
  `P1`, `P2f`, `P3`, `T` are the smoothness energy coefficients");

static PyObject *Py_dp(PyObject *self, PyObject *args, PyObject *kwdict);

PyDoc_STRVAR (Py_data_energy_doc, "data_energy (left, right, [nd=10], [axis=1])\n\
\n\
  Computes the data costs associated with the left/right stereo pair.\n\
  \n\
  `left` and `right` are the left/right rectified stero images\n\
  `nd` is the max absolute value of the disparity\n\
      (will be overwritten if energy is present)\n\
  `axis` is the orientation of the scanlines.\n\
      (scanlines are horizontal if axis=1, vertical otherwise)");
static PyObject *Py_data_energy (PyObject *self, PyObject *args, PyObject *kwdict);

/**
 * Exported method list
 */
static PyMethodDef Methods[] = {
  {"dp", (PyCFunction)Py_dp, METH_KEYWORDS | METH_VARARGS, Py_dp_doc},
  {"data_energy", (PyCFunction)Py_data_energy, METH_KEYWORDS | METH_VARARGS, Py_data_energy_doc},
  {NULL, NULL, 0, NULL}
};
