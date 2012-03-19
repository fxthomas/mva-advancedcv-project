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
 * Computes the first DP passes
 */
static PyObject *imagedp(PyObject *self, PyObject *args, PyObject *kwdict) {
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
  int axis=1;

  // Parse arguments
  static char *kwlist[] = {"left", "right", "backward", "return_point_energy", "nd", "axis", NULL};
  if (!PyArg_ParseTupleAndKeywords (args, kwdict, "O!O!|O!O!ii", kwlist,
        &PyArray_Type, &in1,
        &PyArray_Type, &in2,
        &PyBool_Type, &kwin3,
        &PyBool_Type, &kwin4,
        &nd,
        &axis))
    return NULL;
  if (kwin3) backward = PyObject_IsTrue (kwin3);
  if (kwin4) return_point_energy = PyObject_IsTrue (kwin4);
  if (axis != 0 && axis != 1) {
    PyErr_SetString (PyExc_ValueError, "axis must be 0 (vertical DP) or 1 (horizontal DP)");
    return NULL;
  }

  if (backward) printf ("running simple tree DP (image backward, nd: %d, axis: %d)\n", nd, axis);
  else printf ("running simple tree DP (image forward, nd: %d, axis: %d)\n", nd, axis);

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
    root = (int)(axis==1 ? w : h)-1;
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
  for (scanline = 0; scanline < (axis==1 ? h : w); scanline++) {
    // For each point on the scanline...
    for (p = root; direction*(p-((axis==1 ? w : h)-root-1)) >= 0; p -= direction) {
      // First, compute the values of the energy function for each disparity value
      for (d = -nd; d < nd; d++) {
        if (p-1 < 0 || p+1 >= (axis==1 ? w : h) || p-1+d < 0 || p+1+d >= (axis==1 ? w : h)) continue;
        double i1a = AVAL (left, (axis==1 ? scanline : p-1), (axis==1 ? p-1 : scanline));
        double i1b = AVAL (left, (axis==1 ? scanline : p), (axis==1 ? p : scanline));
        double i1c = AVAL (left, (axis==1 ? scanline : p+1), (axis==1 ? p+1 : scanline));
        double i2a = AVAL (right, (axis==1 ? scanline : p+d-1), (axis==1 ? p+d-1 : scanline));
        double i2b = AVAL (right, (axis==1 ? scanline : p+d), (axis==1 ? p+d : scanline));
        double i2c = AVAL (right, (axis==1 ? scanline : p+d+1), (axis==1 ? p+d+1 : scanline));
        double mpd = disparity (i1a, i1b, i1c, i2a, i2b, i2c); 

        if (return_point_energy) AVAL3 (m, (axis == 1 ? scanline : p), (axis == 1 ? p : scanline), d+nd) = mpd;

        int i;
        double slm = smoothness (-nd, d, i1b, i2b) + AVAL3(F, (axis == 1 ? scanline : p+direction), (axis == 1 ? p+direction : scanline), 0);
        for (i = -nd+1; i < nd; i++) {
          double sln = smoothness (i, d, i1b, i2b) + AVAL3(F, (axis == 1 ? scanline : p+direction), (axis == 1 ? p+direction : scanline), i+nd);
          if (sln < slm) slm = sln;
        }

        AVAL3(F, (axis == 1 ? scanline : p), (axis == 1 ? p : scanline), d+nd) = mpd + slm;
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

/**
 * Computes a DP pass with precomputed pixel energies
 */
static PyObject *dp(PyObject *self, PyObject *args, PyObject *kwdict) {
  ////////////////
  // Initialize //
  ////////////////

  // Declare variables
  PyObject *in1 = 0, *in2 = 0, *in3 = 0, *kwin4 = 0;
  PyArrayObject *energy;
  PyArrayObject *left;
  PyArrayObject *right;
  long h,w;
  int backward = 0, nd = 0;
  int axis = 0;

  // Parse arguments
  static char *kwlist[] = {"left", "right", "energy", "backward", "axis", NULL};
  if (!PyArg_ParseTupleAndKeywords (args, kwdict, "O!O!O!|O!i", kwlist,
        &PyArray_Type, &in1,
        &PyArray_Type, &in2,
        &PyArray_Type, &in3,
        &PyBool_Type, &kwin4,
        &axis))
    return NULL;
  if (kwin4) backward = PyObject_IsTrue (kwin4);
  if (axis != 0 && axis != 1) {
    PyErr_SetString (PyExc_ValueError, "axis must be 0 (vertical DP) or 1 (horizontal DP)");
    return NULL;
  }

  // Convert arrays to contiguous, to be able to access arrays directly
  left = (PyArrayObject*) PyArray_ContiguousFromObject (in1, PyArray_DOUBLE, 2, 2);
  right = (PyArrayObject*) PyArray_ContiguousFromObject (in2, PyArray_DOUBLE, 2, 2);
  energy = (PyArrayObject*) PyArray_ContiguousFromObject (in3, PyArray_DOUBLE, 3, 3);
  if (!energy || !left || !right) return NULL;

  // Store array dimensions for quick access
  h = energy->dimensions[0];
  w = energy->dimensions[1];
  nd = (int)(energy->dimensions[2]/2L);

  // Check if height/width are the same
  if (h != right->dimensions[0] || h != left->dimensions[0] || w != left->dimensions[1] || w != right->dimensions[1]) {
    PyErr_SetString (PyExc_ValueError, "dimensions must be consistent with each other");
    Py_DECREF(left);
    Py_DECREF(right);
    Py_DECREF(energy);
    return NULL;
  }

  if (backward) printf ("running simple tree DP (precomputed backward, nd: %d, axis: %d)\n", nd, axis);
  else printf ("running simple tree DP (precomputed forward, nd: %d, axis: %d)\n", nd, axis);

  ////////////////////
  // Main algorithm //
  ////////////////////
  
  // First DP pass
  int d,scanline,p;
  int direction, root;
  int dimensions[] = {(int)h, (int)w, 2*nd};
  if (backward) {
    direction = 1;
    root = (int)w-1;
  }
  else {
    direction = -1;
    root = 0;
  }

  PyArrayObject *F = (PyArrayObject *)PyArray_FromDims (3, dimensions, PyArray_DOUBLE);

  for (scanline=0; scanline<h; scanline++) for (p=0; p<w; p++) for (d=0; d<2*nd; d++) {
    AVAL3(F, scanline, p, d) = 0;
  }
  
  // For each scanline...
  for (scanline = 0; scanline < (axis==1 ? h : w); scanline++) {
    // For each point on the scanline...
    for (p = root; direction*(p-((axis==1 ? w : h)-root-1)) >= 0; p -= direction) {
      // First, compute the values of the energy function for each disparity value
      for (d = -nd; d < nd; d++) {
        // If we're at the boundary, don't do anything (can't compute)
        if (p-1 < 0 || p+1 >= (axis==1 ? w : h) || p-1+d < 0 || p+1+d >= (axis==1 ? w : h)) continue;

        // Else, recursively perform DP computation
        int i;

        // Per-pixel energy
        double mpd = AVAL3 (energy, (axis == 1 ? scanline : p), (axis == 1 ? p : scanline), d+nd);

        // Pixel values for each image
        double i1b = AVAL (left, (axis == 1 ? scanline : p), (axis == 1 ? p : scanline)); 
        double i2b = AVAL (left, (axis == 1 ? scanline : p+d), (axis == 1 ? p+d : scanline));

        // Neighbor energy
        double slm = smoothness (-nd, d, i1b, i2b) + AVAL3(F, (axis == 1 ? scanline : p+direction), (axis == 1 ? p+direction : scanline), 0);
        for (i = -nd+1; i < nd; i++) {
          double sln = smoothness (i, d, i1b, i2b) + AVAL3(F, (axis == 1 ? scanline : p+direction), (axis == 1 ? p+direction : scanline), i+nd);
          if (sln < slm) slm = sln;
        }

        // The total pixel energy is the sum of the neighbor + per-pixel energy
        AVAL3(F, (axis == 1 ? scanline : p), (axis == 1 ? p : scanline), d+nd) = mpd + slm;
      }
    }
  }

  /////////////
  // Cleanup //
  /////////////

  // Destroy references
  Py_DECREF (energy);
  Py_DECREF (left);
  Py_DECREF (right);

  // Return computed pass
  return (PyObject*) F;
}
