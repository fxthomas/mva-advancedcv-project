/**
 * Base C Source code (simpletree_funcs.c)
 * Created: Sat Mar 17 17:29:12 2012
 *
 * This C source code was developped by François-Xavier Thomas.
 * You are free to copy, adapt or modify it.
 * If you do so, however, leave my name somewhere in the credits, I'd appreciate it ;)
 * 
 * @author François-Xavier Thomas <fx.thomas@gmail.com>
 * @version 1.0
 */

#include "simpletree_funcs.h"

static double P1 = 10.f;
static double P2f = 100.f;
static double P3 = 0.4f;
static double T = 20.f;

double smoothness(double d1, double d2, double i1, double i2) {
  if (d1 == d2) return 0.f;
  else if (d1 == d2-1.f || d2 == d1-1.f) return P1;
  else if (fabs(i1-i2) < T) return P2f*P3;
  else return P2f;
}

double disparity (double i1a, double i1b, double i1c, double i2a, double i2b, double i2c) {
  // Compute dh (x_i, y_i, L, R)
  double i2m = 0.5f * (i2a + i2b);
  double i2p = 0.5f * (i2c + i2b);
  double i2min = MIN (MIN(i2m, i2p), i2b);
  double i2max = MAX (MAX(i2m, i2p), i2b);
  double i2dh = MAX (MAX (i1b - i2max, i2min - i1b), 0);

  // Compute dh (y_i, x_i, R, L)
  double i1m = 0.5f * (i1a + i1b);
  double i1p = 0.5f * (i1c + i1b);
  double i1min = MIN (MIN(i1m, i1p), i1b);
  double i1max = MAX (MAX(i1m, i1p), i1b);
  double i1dh = MAX (MAX (i2b - i1max, i1min - i2b), 0);

  return MIN (i1dh, i2dh);
}

PyArrayObject *data_energy (PyArrayObject *left, PyArrayObject *right, int nd, int axis) {
  // Store array dimensions for quick access
  int h = (int)(left->dimensions[0]);
  int w = (int)(left->dimensions[1]);

  // Prepare variables
  int dimensions[] = {h, w, 2*nd};
  int d,scanline,p;
  PyArrayObject *m = (PyArrayObject *)PyArray_FromDims (3, dimensions, PyArray_DOUBLE);

  // For each scanline...
  for (scanline = 0; scanline < (axis==1 ? h : w); scanline++) {
    // For each point on the scanline...
    for (p = 0; p < (axis==1 ? w : h); p++) {
      // First, compute the values of the energy function for each disparity value
      for (d = -nd; d < nd; d++) {
        double mpd;

        // On the border, set the energy to 0
        if (p-1 < 0 || p+1 >= (axis==1 ? w : h) || p-1+d < 0 || p+1+d >= (axis==1 ? w : h)) {
          mpd = 0;

        // Else, compute it according to paper "Depth Discontinuities by Pixel-to-Pixel Stereo"
        } else {
          double i1a = AVAL (left, (axis==1 ? scanline : p-1), (axis==1 ? p-1 : scanline));
          double i1b = AVAL (left, (axis==1 ? scanline : p), (axis==1 ? p : scanline));
          double i1c = AVAL (left, (axis==1 ? scanline : p+1), (axis==1 ? p+1 : scanline));
          double i2a = AVAL (right, (axis==1 ? scanline : p+d-1), (axis==1 ? p+d-1 : scanline));
          double i2b = AVAL (right, (axis==1 ? scanline : p+d), (axis==1 ? p+d : scanline));
          double i2c = AVAL (right, (axis==1 ? scanline : p+d+1), (axis==1 ? p+d+1 : scanline));
          mpd = disparity (i1a, i1b, i1c, i2a, i2b, i2c); 
        }

        AVAL3 (m, (axis == 1 ? scanline : p), (axis == 1 ? p : scanline), d+nd) = mpd;
      }
    }
  }


  return m;
}

PyArrayObject *dp (PyArrayObject *left, PyArrayObject *right, PyArrayObject *energy, int backward, int nd, int axis) {
  // Store array dimensions for quick access
  int h = (int)(left->dimensions[0]);
  int w = (int)(left->dimensions[1]);

  // First DP pass
  int dimensions[] = {h, w, 2*nd};
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

  return F;
}
