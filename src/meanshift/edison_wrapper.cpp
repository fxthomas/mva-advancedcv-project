/**
 * Base C Source code (edison_wrapper.c)
 * Created: Mon Mar 26 17:11:15 2012
 *
 * This C source code was developped by François-Xavier Thomas.
 * You are free to copy, adapt or modify it.
 * If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
 * 
 * @author François-Xavier Thomas <fx.thomas@gmail.com>
 * @version 1.0
 */

#include "edison_wrapper.h"

PyMODINIT_FUNC initedison (void) {
  if (!Py_InitModule ("edison", Methods)) return;
  import_array();
}

static PyObject *Py_meanshift (PyObject *self, PyObject *args, PyObject *kwdict) {
  // Python objects
  PyObject *in_rgbimage = 0, *in_fimage = 0, *inb_synergistic = 0;
  PyArrayObject *rgbimage = 0, *fimage = 0;

  // Default algorithm parameters
  int steps = 3,
      synergistic = 0,
      SpatialBandWidth = 7,
      MinimumRegionArea = 20,
      SpeedUp_int = 1,
      GradientWindowRadius = 2;
  float RangeBandWidth = 6.5,
        MixtureParameter = 0.3,
        EdgeStrengthThreshold = 0.3;
  enum SpeedUpLevel SpeedUp;

  // Parse arguments
  static char *kwlist[] = {"fimage", "rgbimage", "steps", "synergistic", "SpatialBandWidth", "RangeBandWidth", "MinimumRegionArea", "SpeedUp", "GradientWindowRadius", "MixtureParameter", "EdgeStrengthThreshold"};
  if (!PyArg_ParseTupleAndKeywords (args, kwdict, "O!|O!iO!fiiiff", kwlist,
        &PyArray_Type, &in_fimage,
        &PyArray_Type, &in_rgbimage,
        &steps,
        &PyBool_Type, &inb_synergistic,
        &SpatialBandWidth,
        &RangeBandWidth,
        &MinimumRegionArea,
        &SpeedUp,
        &GradientWindowRadius,
        &MixtureParameter,
        &EdgeStrengthThreshold)) return NULL;

  // Steps
  if (steps < 1 || steps > 3) {
    PyErr_SetString (PyExc_ValueError, "steps must be equal to 1, 2 or 3");
    return NULL;
  }

  // Is this a synergistic segmentation?
  synergistic = inb_synergistic && PyObject_IsTrue (inb_synergistic);
  if (synergistic) fprintf (stderr, "warning: synergistic segmentation not implemented, see original implementation for details");

  // Speedup value
  switch (SpeedUp_int) {
    case 0:
      SpeedUp = NO_SPEEDUP;
      break;
    case 1:
      SpeedUp = MED_SPEEDUP;
      break;
    case 2:
      SpeedUp = HIGH_SPEEDUP;
      break;
    default:
      PyErr_SetString (PyExc_ValueError, "speedup must be equal to 0, 1, or 2");
      return NULL;
  }

  // Test if images were correctly loaded
  
  // Transform them into contiguous arrays
  fimage = (PyArrayObject *)PyArray_ContiguousFromObject (in_fimage, PyArray_FLOAT, 3, 3);
  if (in_rgbimage) rgbimage = (PyArrayObject *)PyArray_ContiguousFromObject (in_rgbimage, PyArray_FLOAT, 3, 3);

  if (!fimage || (in_rgbimage && !in_rgbimage)) {
    PyErr_SetString (PyExc_ValueError, "could not create contiguous arrays from RGB and converted images");
    return NULL;
  }

  // Test image dimensions
  int h = (int)(fimage->dimensions[0]);
  int w = (int)(fimage->dimensions[1]);
  int N = (int)(fimage->dimensions[2]);
  if (rgbimage) {
    if (h != (int)(rgbimage->dimensions[0]) || w != (int)(rgbimage->dimensions[1]) || N != (int)(rgbimage->dimensions[2])) {
      PyErr_SetString (PyExc_ValueError, "converted and RGB images must have the same shape");

      Py_DECREF (rgbimage);
      Py_DECREF (fimage);
      return NULL;
    }
  }

  /******************
   * Main algorithm *
   ******************/

  // Load image
  msImageProcessor ms;
  ms.DefineLInput ((float*)(fimage->data), h, w, N);
  if (ms.ErrorStatus) {
    PyErr_SetString (PyExc_ValueError, ms.ErrorMessage);

    if (rgbimage) Py_DECREF (rgbimage);
    Py_DECREF (fimage);
    return NULL;
  }

  // Load kernel
  kernelType k[2] = {DefualtKernelType, DefualtKernelType};
  int P[2] = {DefualtSpatialDimensionality, N};
  float tempH[2] = {1.f, 1.f};
  ms.DefineKernel (k, tempH, P, 2);
  if (ms.ErrorStatus) {
    PyErr_SetString (PyExc_ValueError, ms.ErrorMessage);

    if (rgbimage) Py_DECREF (rgbimage);
    Py_DECREF (fimage);
    return NULL;
  }

  /* Not implemented
   * See original implementation
  float *conf = NULL;
  float *grad = NULL;
  float *wght = NULL;

  if (synergistic) {
  }
  */

  // Do it!
  ms.Filter (SpatialBandWidth, RangeBandWidth, SpeedUp);
  if (ms.ErrorStatus) {
    PyErr_SetString (PyExc_ValueError, ms.ErrorMessage);

    if (rgbimage) Py_DECREF (rgbimage);
    Py_DECREF (fimage);
    return NULL;
  }

  if (steps == 2) {
    ms.FuseRegions (RangeBandWidth, MinimumRegionArea);
    if (ms.ErrorStatus) {
      PyErr_SetString (PyExc_ValueError, ms.ErrorMessage);

      if (rgbimage) Py_DECREF (rgbimage);
      Py_DECREF (fimage);
      return NULL;
    }
  }

  // Get output data
  int dimensions[] = {h, w, N};
  int *labels, *count;
  float *modes;
  PyArrayObject *out_fimage = (PyArrayObject *)PyArray_FromDims (3, dimensions, PyArray_FLOAT);
  ms.GetRawData ((float*)(out_fimage->data));
  int RegionCount = ms.GetRegions (&labels, &modes, &count);

  // Labels
  int labeldim[] = {h, w};
  PyArrayObject *out_labels = (PyArrayObject *)PyArray_FromDims (2, labeldim, PyArray_INT);
  for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) AVALI(out_labels, i, j) = labels[i*w + j];
  delete[] labels;

  // Modes
  int modedim[] = {N, RegionCount};
  PyArrayObject *out_modes = (PyArrayObject *)PyArray_FromDims (2, modedim, PyArray_FLOAT);
  for (int i = 0; i < N; i++) for (int j = 0; j < RegionCount; j++) AVAL (out_modes, i, j) = modes[i*RegionCount + j];
  delete[] modes;

  // Return a tuple : (out_fimage, out_labels, out_modes)
  PyObject *ret = PyTuple_New (3);
  PyTuple_SetItem (ret, 0, (PyObject*)out_fimage);
  PyTuple_SetItem (ret, 1, (PyObject*)out_labels);
  PyTuple_SetItem (ret, 2, (PyObject*)out_modes);

  if (rgbimage) Py_DECREF (rgbimage);
  Py_DECREF (fimage);
  return ret;
}
