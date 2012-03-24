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

PyArrayObject *data_energy (PyArrayObject *left, PyArrayObject *right, int nd, int axis) {
  // Store array dimensions for quick access
  int h = (int)(left->dimensions[0]);
  int w = (int)(left->dimensions[1]);

  // Prepare variables
  int dimensions[] = {h, w, 2*nd};
  int d,scanline,p;
  PyArrayObject *m = (PyArrayObject *)PyArray_FromDims (3, dimensions, PyArray_DOUBLE);

  // For each scanline...
  for (scanline = nd+1; scanline < (axis==1 ? h : w) - (nd+1); scanline++) {
    // For each point on the scanline...
    for (p = nd+1; p < (axis==1 ? w : h) - nd-1; p++) {
      // First, compute the values of the energy function for each disparity value
      for (d = -nd; d < nd; d++) {
        // Pixel values
        double i1a = AVAL (left, (axis==1 ? scanline : p-1), (axis==1 ? p-1 : scanline));
        double i1b = AVAL (left, (axis==1 ? scanline : p), (axis==1 ? p : scanline));
        double i1c = AVAL (left, (axis==1 ? scanline : p+1), (axis==1 ? p+1 : scanline));
        double i2a = AVAL (right, (axis==1 ? scanline : p+d-1), (axis==1 ? p+d-1 : scanline));
        double i2b = AVAL (right, (axis==1 ? scanline : p+d), (axis==1 ? p+d : scanline));
        double i2c = AVAL (right, (axis==1 ? scanline : p+d+1), (axis==1 ? p+d+1 : scanline));

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

        AVAL3 (m, (axis == 1 ? scanline : p), (axis == 1 ? p : scanline), d+nd) = MIN (i1dh, i2dh);
      }
    }
  }


  return m;
}

PyArrayObject *dp (PyArrayObject *left, PyArrayObject *right, PyArrayObject *energy, int backward, int nd, int axis, double P1, double P2f, double P3, double T) {
  // Store array dimensions for quick access
  int h = (int)(left->dimensions[0]);
  int w = (int)(left->dimensions[1]);
  int dimensions[] = {h, w, 2*nd};
  int dim_lmin[] = {h, w};
  int d, scanline, p, direction, root;

  // First DP pass
  if (backward) {
    direction = 1;
    root = (int)(axis==1 ? w : h)-1-(nd+1);
  }
  else {
    direction = -1;
    root = nd+1;
  }

  // Create array
  PyArrayObject *F = (PyArrayObject *)PyArray_FromDims (3, dimensions, PyArray_DOUBLE);
  PyArrayObject *LMin = (PyArrayObject *)PyArray_FromDims (2, dim_lmin, PyArray_DOUBLE);

  // For each scanline...
  for (scanline = nd+1; scanline < (axis==1 ? h : w) - (nd+1); scanline++) {
    // For each point on the scanline...
    for (p = root; direction*(p-((axis==1 ? w : h)-root-1)) >= 0; p -= direction) {
      // Point coordinates
      int pi = (axis == 1 ? scanline : p);
      int pj = (axis == 1 ? p : scanline);
      int qi = (axis == 1 ? scanline : p + direction);
      int qj = (axis == 1 ? p + direction : scanline);
      AVAL (LMin, pi, pj) = INFINITY;

      // Pixel values for each image
      double i1b = AVAL (left, qi, qj); 
      double i2b = AVAL (right, qi, qj);

      // Smoothness coefficient
      double P2 = P2f; if (fabs (i1b - i2b) < T) P2 *= P3;

      // Compute the values of the energy function for each disparity value
      for (d = -nd; d < nd; d++) {

        // Per-pixel energy
        double mpd = AVAL3 (energy, pi, pj, d+nd);

        // Neighbor energy
        double sl_curr = 0;
        if (p != root) {
          double lqd = AVAL3 (F, qi, qj, d+nd);
          double lqdm = AVAL3 (F, qi, qj, d+nd-1) + P1;
          double lqdp = AVAL3 (F, qi, qj, d+nd+1) + P1;
          double min_lqd = AVAL (LMin, qi, qj) + P2;

          sl_curr = MIN (lqd, min_lqd);
          if (d > -nd) sl_curr = MIN (sl_curr, lqdm);
          if (d < nd-1) sl_curr = MIN (sl_curr, lqdp);
        }

        // Value of l'(p,d)
        double lpd = mpd + sl_curr;
        if (lpd < AVAL (LMin, pi, pj)) AVAL (LMin, pi, pj) = lpd;

        // The total pixel energy is the sum of the neighbor + per-pixel energy
        AVAL3(F, pi, pj, d+nd) = lpd;
      }
    }
  }

  Py_DECREF (LMin);

  return F;
}
