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
static PyObject *Py_dp (PyObject *self, PyObject *args, PyObject *kwdict) {
  ////////////////
  // Initialize //
  ////////////////

  // Declare variables
  PyObject *in1 = 0, *in2 = 0, *kwin3 = 0, *kwin4 = 0;
  PyArrayObject *left = 0;
  PyArrayObject *right = 0;
  PyArrayObject *energy = 0;
  int backward = 0;
  int return_point_energy = 0;
  int nd = 10;
  int axis=1;

  // Parse arguments
  static char *kwlist[] = {"left", "right", "energy", "backward", "nd", "axis", NULL};
  if (!PyArg_ParseTupleAndKeywords (args, kwdict, "O!O!|O!O!ii", kwlist,
        &PyArray_Type, &in1,
        &PyArray_Type, &in2,
        &PyArray_Type, &kwin3,
        &PyBool_Type, &kwin4,
        &nd,
        &axis))
    return NULL;

  // Is this a backward or a forward pass?
  if (kwin4) backward = PyObject_IsTrue (kwin4);

  // Is the axis valid?
  //   0: Vertical scanlines
  //   1: Horizontal scanlines
  if (axis != 0 && axis != 1) {
    PyErr_SetString (PyExc_ValueError, "axis must be 0 (vertical DP) or 1 (horizontal DP)");
    return NULL;
  }

  // Prepare left and right images (we need contiguous arrays to ba able to read them correctly from C
  left = (PyArrayObject*) PyArray_ContiguousFromObject (in1, PyArray_DOUBLE, 2, 2);
  right = (PyArrayObject*) PyArray_ContiguousFromObject (in2, PyArray_DOUBLE, 2, 2);
  if (!left || !right) return NULL;

  // Check if height/width are the same
  if (left->dimensions[0] != right->dimensions[0] || left->dimensions[1] != right->dimensions[1]) {
    PyErr_SetString (PyExc_ValueError, "arrays must have the same dimension");
    Py_DECREF(left);
    Py_DECREF(right);
    return NULL;
  }

  // If we have the 'energy' parameter...
  if (kwin3) {
    // Prepare the contiguous array
    energy = (PyArrayObject*) PyArray_ContiguousFromObject (kwin3, PyArray_DOUBLE, 3, 3);

    // We don't need to return it
    return_point_energy = 0;

    // Check if first dimensions are the same as left/right images
    if (energy->dimensions[0] != left->dimensions[0] || energy->dimensions[1] != left->dimensions[1]) {
      PyErr_SetString (PyExc_ValueError, "energy must have the same dimension as left and right images");
      Py_DECREF(left);
      Py_DECREF(right);
      Py_DECREF(energy);
      return NULL;
    }

    // The last dimension is 2 times the number of disparity increments
    nd = (int)(energy->dimensions[2])/2;

  // If we don't have the `energy` parameter...
  } else {
    // Generate image-based per-pixel energy
    energy = data_energy (left, right, nd, axis);

    // We return this energy array at the end, so that it can be used for further computations
    return_point_energy = 1;
  }

  ////////////////////
  // Main algorithm //
  ////////////////////
  
  if (backward) printf ("running simple tree DP (backward, nd: %d, axis: %d)\n", nd, axis);
  else printf ("running simple tree DP (forward, nd: %d, axis: %d)\n", nd, axis);
  PyArrayObject *F = dp (left, right, energy, backward, nd, axis);

  /////////////
  // Cleanup //
  /////////////

  // Destroy references
  Py_DECREF (left);
  Py_DECREF (right);
  
  if (return_point_energy) {
    PyObject *ret = PyTuple_New (2);
    PyTuple_SetItem (ret, 0, (PyObject*)F);
    PyTuple_SetItem (ret, 1, (PyObject*)energy);

    return ret;
  } else {
    Py_DECREF (energy);
    return (PyObject*) F;
  }
}
