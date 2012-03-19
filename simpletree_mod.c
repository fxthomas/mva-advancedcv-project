/**
 * Base C Source code (simpletree.c)
 * Created: Sat Mar 17 15:20:43 2012
 *
 * This C source code was developped by François-Xavier Thomas.
 * You are free to copy, adapt or modify it.
 * If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
 * 
 * @author François-Xavier Thomas <fx.thomas@gmail.com>
 * @version 1.0
 */

#include "simpletree_mod.h"
#include "simpletree_funcs.h"
#include <stdio.h>

/**
 * Initialize module
 */
PyMODINIT_FUNC initsimpletree (void) {
  if (!Py_InitModule ("simpletree", Methods)) return;
  import_array();
}

/**
 * Disparity map computation function
 */
static PyObject *dp_pass(PyObject *self, PyObject *args, PyObject *kwdict) {
  ////////////////
  // Initialize //
  ////////////////

  // Declare variables
  PyObject *in1 = 0, *in2 = 0, *kwin3 = 0, *kwin4 = 0;
  PyArrayObject *left;
  PyArrayObject *right;
  long h,w;
  int backward = 0;
  int return_point_energy = 0;
  int nd = 10;

  // Parse arguments
  static char *kwlist[] = {"left", "right", "backward", "return_point_energy", "nd", NULL};
  if (!PyArg_ParseTupleAndKeywords (args, kwdict, "O!O!|O!O!i", kwlist,
        &PyArray_Type, &in1,
        &PyArray_Type, &in2,
        &PyBool_Type, &kwin3,
        &PyBool_Type, &kwin4,
        &nd))
    return NULL;
  if (kwin3) backward = PyObject_IsTrue (kwin3);
  if (kwin4) return_point_energy = PyObject_IsTrue (kwin4);

  if (backward) printf ("running simple tree (backward, nd: %d)\n", nd);
  else printf ("running simple tree (forward, nd: %d)\n", nd);

  // Convert arrays to contiguous, to be able to access arrays directly
  left = (PyArrayObject*) PyArray_ContiguousFromObject (in1, PyArray_DOUBLE, 2, 2);
  right = (PyArrayObject*) PyArray_ContiguousFromObject (in2, PyArray_DOUBLE, 2, 2);
  if (!left || !right) return NULL;

  // Store array dimensions for quick access
  h = left->dimensions[0];
  w = left->dimensions[1];

  // Check if height/width are the same
  if (h != right->dimensions[0] || w != right->dimensions[1]) {
    PyErr_SetString (PyExc_ValueError, "arrays must have the same dimension");
    Py_DECREF(left);
    Py_DECREF(right);
    return NULL;
  }

  ////////////////////
  // Main algorithm //
  ////////////////////
  
  // First DP pass
  int dimensions[] = {(int)h, (int)w, 2*nd};
  int d,scanline,p;
  int direction, root;
  if (backward) {
    direction = 1;
    root = (int)w-1;
  }
  else {
    direction = -1;
    root = 0;
  }

  PyArrayObject *F = (PyArrayObject *)PyArray_FromDims (3, dimensions, PyArray_DOUBLE);
  PyArrayObject *m = 0;
  if (return_point_energy) m = (PyArrayObject *)PyArray_FromDims (3, dimensions, PyArray_DOUBLE);

  for (scanline=0; scanline<h; scanline++) for (p=0; p<w; p++) for (d=0; d<2*nd; d++) {
    AVAL3(F, scanline, p, d) = 0;
  }
  
  // For each scanline...
  for (scanline = 0; scanline < h; scanline++) {
    // For each point on the scanline...
    for (p = root; direction*(p-(w-root-1)) >= 0; p -= direction) {
      // First, compute the values of the energy function for each disparity value
      for (d = -nd; d < nd; d++) {
        if (p-1 < 0 || p+1 >= w || p-1+d < 0 || p+1+d >= w) continue;
        double i1a = AVAL (left, scanline, p - 1);
        double i1b = AVAL (left, scanline, p);
        double i1c = AVAL (left, scanline, p + 1);
        double i2a = AVAL (right, scanline, p+d - 1);
        double i2b = AVAL (right, scanline, p+d);
        double i2c = AVAL (right, scanline, p+d + 1);
        double mpd = disparity (i1a, i1b, i1c, i2a, i2b, i2c); 

        if (return_point_energy) AVAL3 (m, scanline, p, d+nd) = mpd;

        int i;
        double slm = smoothness (-nd, d, i1b, i2b) + AVAL3(F, scanline, p+direction, 0);
        for (i = -nd+1; i < nd; i++) {
          double sln = smoothness (i, d, i1b, i2b) + AVAL3(F, scanline, p+direction, i+nd);
          if (sln < slm) slm = sln;
        }

        AVAL3(F, scanline, p, d+nd) = mpd + slm;
      }
    }
  }

  /////////////
  // Cleanup //
  /////////////

  // Destroy references
  Py_DECREF (left);
  Py_DECREF (right);
  
  if (return_point_energy) {
    PyObject *ret = PyTuple_New (2);
    PyTuple_SetItem (ret, 0, (PyObject*)F);
    PyTuple_SetItem (ret, 1, (PyObject*)m);

    // Return F and m
    return ret;
  } else {
    return (PyObject*) F;
  }
}
