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
