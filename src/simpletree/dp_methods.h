/**
 * Base C Header (dp_methods.h)
 * Created: Sat Mar 17 17:29:15 2012
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
#define NO_IMPORT_ARRAY

#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define AVAL(A,I,J) (*(double*)(A->data + (I)*(A->strides[0]) + (J)*(A->strides[1])))
#define AVAL3(A,I,J,K) (*(double*)(A->data + (I)*(A->strides[0]) + (J)*(A->strides[1]) + (K)*(A->strides[2])))

/**
 * Return per-pixel energy array
 */
PyArrayObject *data_energy (PyArrayObject *left, PyArrayObject *right, int nd, int axis);

/**
 * This function does the actual DP computation
 */
PyArrayObject *dp (PyArrayObject *left, PyArrayObject *right, PyArrayObject *energy, int backward, int nd, int axis, double P1, double P2f, double P3, double T);
