/**
 * Base C Header (edison_wrapper.h)
 * Created: Mon Mar 26 17:11:25 2012
 *
 * This C Header was developped by François-Xavier Thomas.
 * You are free to copy, adapt or modify it.
 * If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
 * 
 * @author François-Xavier Thomas <fx.thomas@gmail.com>
 * @version 1.0
 */

#define PY_ARRAY_UNIQUE_SYMBOL __PyArrayEdison
#define AVAL(A,I,J) (*(float*)(A->data + (I)*(A->strides[0]) + (J)*(A->strides[1])))
#define AVALI(A,I,J) (*(int*)(A->data + (I)*(A->strides[0]) + (J)*(A->strides[1])))
#define AVAL3(A,I,J,K) (*(float*)(A->data + (I)*(A->strides[0]) + (J)*(A->strides[1]) + (K)*(A->strides[2])))

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

#include "edison/segm/msImageProcessor.h"

#include "edison/edge/BgImage.h"
#include "edison/edge/BgDefaults.h"
#include "edison/edge/BgEdge.h"
#include "edison/edge/BgEdgeList.h"
#include "edison/edge/BgEdgeDetect.h"


const kernelType DefualtKernelType = Uniform;
const unsigned int DefualtSpatialDimensionality = 2;
bool CmCDisplayProgress = false; /* disable display promt */

/**
 * Module initialization
 */
PyMODINIT_FUNC initedison (void);

/**
 * Declare module methods
 */
PyDoc_STRVAR (Py_meanshift_doc, "meanshift (image, [SpatialBandwidth=7], [RangeBandWidth=6.5], [SpeedUp=1], [MinimumRegionArea=])\n\
    \n\
    Computes the Mean-Shift segmentation for the given image,\n\
    as described in the paper \"Mean Shift: A robust approach toward featurespace analysis.\"\n\
    by D. Comanicu and P.Meer.\n\
    \n\
    This is a simple wrapper around the EDISON source, which can be found here :\n\
      http://coewww.rutgers.edu/riul/research/code/EDISON/index.html\n\
    \n\
    Based on the work of Shawn Lankton (http://www.shawnlankton.com/2007/11/mean-shift-segmentation-in-matlab/)\n\
    and Shai Bagon (http://www.wisdom.weizmann.ac.il/~bagon/matlab.html).\n\
    \n\
    Arguments:\n\
     `image` : The input image for the segmentation algorithm (LUV colorspace is better, the paper says)\n\
        This image *must* be of type np.float32, else it will raise a ValueError (limitation of EDISON)\n\
     `SpatialBandWidth`, `RangeBandWidth`, `SpeedUp`, `MinimumRegionArea` are parameters of the EDISON system,\n\
        look at the docs on the EDISON website for more information.\n\
    \n\
    Return values: A tuple (out_fimage, labels, modes), where:\n\
      `out_fimage` : The segmented image\n\
      `labels` : The different labels used in the segmentation\n\
      `modes` : The average pixel value of each label");

static PyObject *Py_meanshift (PyObject *self, PyObject *args, PyObject *kwdict);

/**
 * Exported method list
 */
static PyMethodDef Methods[] = {
  {"meanshift", (PyCFunction)Py_meanshift, METH_KEYWORDS | METH_VARARGS, Py_meanshift_doc},
  {NULL, NULL, 0, NULL}
};
