/* Production: laminar vortex shedding in the shallow channel, 2-D NS in the dye plane.
   Nondimensional: D = 1, U (mean inflow) = 1, nu = 1/Re. World (m) = 0.01 * (x, y); t_phys = t * D/U = t/1.5 s.
   Channel x in [-30,30], walls y = +-6, cylinder centre (-4, 0). 7 needle sources at x = -9.3,
   y = -2.4..2.4 step 0.8, colour index [0,1,2,0,1,2,0] (lower y first, = scene_params injectors.y order).
   Usage: ./prod LEVEL TREL_NOM PHI TTRAV NOUT_EVERY [restart_dump]
     precursor runs dye-free; at TREL_NOM the release time is shifted forward so that
     t_rel + TTRAV = (Cl upward zero crossing) + PHI*T; t0 = first frame time (multiple of 1/16) after the
     dye front (max x where c0+c1+c2 > 0.25) reaches x_cyl + 0.6 D = -3.4.  Then 241 frames, dt_f = 1/16. */
#include "embed.h"
#include "navier-stokes/centered.h"
#include "tracer.h"
#include "cylforce.h"

scalar c0[], c1[], c2[];
scalar * tracers = {c0, c1, c2};
double Re = 150.;
int maxlevel = 10, nout_every = 1;
double trel_nom = 40., phi = 0., ttrav = 6.0;
double t_rel = -1., t_front = -1., t0 = -1.;
const double XC = -4., YC = 0., W2 = 6., XN = -9.3, XFRONT = -3.4, DTF = 1./16.;
double LAMBDA = 6.17e-3*0.01/0.015;  // B: lambda = (pi^2/4) nu/h^2 = 6.17e-3 1/s, nondim lambda D/U
int NOCYL = 0;
double TV0 = 1e30, VX0 = -10.; int NVX = 800;  /* v003: VFULL=1 -> full channel x in [-30,30], 1200 cells */
int PROFILE = 0;        // 0 plug (B), 1 parabolic
double UIN = 0.01352/0.015; // B: plug U_in = 0.01352 m/s, U_ref = 0.015 m/s centreline at x_cyl
char * restart = NULL;
face vector muv[], av[];
FILE * flog, * fev;
#define NMAX 200000
double tl[NMAX], cl[NMAX]; int nl = 0;

double uin (double y) {
  if (fabs(y) >= W2) return 0.;
  return PROFILE == 1 ? 1.5*UIN*(1. - sq(y/W2)) : UIN;
}

int main (int argc, char * argv[])
{
  if (argc > 1) maxlevel = atoi (argv[1]);
  if (argc > 2) trel_nom = atof (argv[2]);
  if (argc > 3) phi = atof (argv[3]);
  if (argc > 4) ttrav = atof (argv[4]);
  if (argc > 5) nout_every = atoi (argv[5]);
  if (argc > 6) restart = argv[6];
  if (getenv ("LAMBDA")) LAMBDA = atof (getenv ("LAMBDA"));
  if (getenv ("PROFILE")) PROFILE = atoi (getenv ("PROFILE"));
  if (getenv ("NOCYL")) NOCYL = 1;
  if (getenv ("TV0")) TV0 = atof (getenv ("TV0"));
  if (getenv ("VFULL")) { VX0 = -30.; NVX = 1200; }
  if (getenv ("UIN")) UIN = atof (getenv ("UIN"));
  L0 = 60.;
  origin (-30., -30. - 0.0073);   // walls not on grid lines
  N = 128;
  mu = muv;
  a = av;
  TOLERANCE = 1e-4;
  flog = fopen ("forces.dat", restart ? "a" : "w");
  fev = fopen ("events.txt", restart ? "a" : "w");
  run();
}

event properties (i++) {
  foreach_face()
    muv.x[] = fm.x[]/Re;
}

u.n[left]  = dirichlet (uin(y));
u.t[left]  = dirichlet (0.);
p[left]    = neumann (0.);
pf[left]   = neumann (0.);
c0[left] = dirichlet (0.); c1[left] = dirichlet (0.); c2[left] = dirichlet (0.);
u.n[right] = neumann (0.);
p[right]   = dirichlet (0.);
pf[right]  = dirichlet (0.);
u.n[embed] = dirichlet (0.);
u.t[embed] = dirichlet (0.);

double geom (double x, double y) {
  if (NOCYL) return min (W2 - y, W2 + y);
  return min (min (W2 - y, W2 + y), sqrt (sq(x - XC) + sq(y - YC)) - 0.5);
}

event init (t = 0) {
  if (restart) {
    if (!restore (file = restart)) { fprintf (stderr, "cannot restore %s\n", restart); exit (1); }
    solid (cs, fs, geom(x,y));
    fprintf (fev, "restart %s t=%g\n", restart, t);
    if (getenv ("NODYE")) { foreach() c0[] = c1[] = c2[] = 0.; fprintf (fev, "tracers cleared (NODYE)\n"); }
    return 0;
  }
  refine (fabs(y) < W2 + 0.5 && level < maxlevel - 3);
  refine (sq(x - XC) + sq(y - YC) < sq(0.8) && level < maxlevel);
  solid (cs, fs, geom(x,y));
  /* asymmetric kick: one Lamb-Oseen vortex (Gamma = 1, core 0.5) behind the cylinder at (-2.5, 0.6) */
  foreach() {
    double dx = x + 2.5, dy = y - 0.6, r2 = sq(dx) + sq(dy) + 1e-12;
    double ut = 1./(2.*pi*r2)*(1. - exp(-r2/0.25));
    u.x[] = cs[] ? uin(y) - ut*dy : 0.;
    u.y[] = cs[] ? ut*dx : 0.;
    if (NOCYL) { u.x[] = cs[] ? uin(y) : 0.; u.y[] = 0.; }
  }
}

event acceleration (i++) {
  foreach_face()
    av.x[] = - LAMBDA*face_value (u.x, 0);
}

/* sources: hold c = max(c, gaussian) at each needle tip once released */
event sources (i++) {
  if (t_rel < 0. || t < t_rel) return 0;
  double r0 = 0.035, dl = 60./(1 << maxlevel), sig = max (r0, dl);
  foreach() {
    if (fabs(x - XN) < 4.*sig + Delta && fabs(y) < 2.4 + 4.*sig + Delta)
      for (int k = 0; k < 7; k++) {
	double yk = -2.4 + 0.8*k;
	double g = clamp ((max (sig, 0.5*Delta) - sqrt (sq(x - XN) + sq(y - yk)))/Delta + 0.5, 0., 1.);
	int cidx = k % 3;
	if (cidx == 0 && g > c0[]) c0[] = g;
	if (cidx == 1 && g > c1[]) c1[] = g;
	if (cidx == 2 && g > c2[]) c2[] = g;
      }
  }
}

void plan_release () {
  if (getenv ("T0_FIXED")) {
    t0 = atof (getenv ("T0_FIXED")); t_rel = t0 - ttrav;
    fprintf (fev, "plan: fixed (from earlier history) target_t0=%g t_rel=%g ttrav=%g\n", t0, t_rel, ttrav);
    fflush (fev); return;
  }
  /* upward zero crossings of Cl (over the history) */
  double cr[2000]; int nc = 0;
  for (int k = 1; k < nl && nc < 2000; k++)
    if (cl[k-1] < 0. && cl[k] >= 0. && tl[k] > 10.)
      cr[nc++] = tl[k-1] + (tl[k] - tl[k-1])*(-cl[k-1])/(cl[k] - cl[k-1]);
  if (nc < 3) { t_rel = t; fprintf (fev, "plan: too few crossings (%d), release now t=%g\n", nc, t); return; }
  double T = (cr[nc-1] - cr[max(0, nc-4)])/(nc - 1 - max(0, nc-4));
  FILE * fphi = fopen ("phi.txt", "r");   /* phase can be set late (measurement anchor) */
  if (fphi) { if (fscanf (fphi, "%lf", &phi) != 1) phi = 0.; fclose (fphi); fprintf (fev, "phi from phi.txt = %g\n", phi); }
  double target = cr[nc-1] + phi*T;
  while (target - ttrav < t + 0.05) target += T;
  target = DTF*floor (target/DTF + 0.5);
  t0 = target;
  t_rel = target - ttrav;
  fprintf (fev, "plan: T=%g last_cross=%g phi=%g ttrav=%g target_t0=%g t_rel=%g\n",
	   T, cr[nc-1], phi, ttrav, target, t_rel);
  fflush (fev);
}

event logfile (i++) {
  coord Fp, Fmu;
  cyl_force (p, u, mu, XC, YC, 0.8, &Fp, &Fmu);
  double Cd = 2.*(Fp.x + Fmu.x), Cl = 2.*(Fp.y + Fmu.y);
  if (nl < NMAX) { tl[nl] = t; cl[nl] = Cl; nl++; }
  double vp = interpolate (u.y, XC + 3., 0.), uc = interpolate (u.x, XC, 0.);
  double front = -30.;
  if (t_rel >= 0.)
    foreach (reduction(max:front))
      if (cs[] > 0.5 && c0[] + c1[] + c2[] > 0.25 && x > front) front = x;
  double m0 = 0., m1 = 0., m2 = 0.;
  if (t_rel >= 0.)
    foreach (reduction(+:m0) reduction(+:m1) reduction(+:m2)) {
      double dv = cm[]*sq(Delta);
      m0 += c0[]*dv; m1 += c1[]*dv; m2 += c2[]*dv;
    }
  fprintf (flog, "%.6f %.6f %.6f %.6f %.4f %ld %d %d %.5g %.5g %.5g %.6f\n", t, Cd, Cl, vp, front,
	   grid->tn, mgp.i, mgu.i, m0, m1, m2, uc);
  if (i % 50 == 0) fflush (flog);
  if (t_rel < 0. && t >= trel_nom) plan_release ();
  if (t_rel >= 0. && t_front < 0. && front >= XFRONT) {
    t_front = t;
    fprintf (fev, "front reached x=%g at t=%g ; t0=%g ; t_rel=%g ; actual travel=%g\n",
	     front, t, t0, t_rel, t - t_rel);
    fflush (fev);
  }
}

/* periodic dumps for phase-shift reruns (last 8 time units before the nominal release) */
event dumps (t += 0.5) {
  if (t >= trel_nom - 8. && t <= trel_nom + 0.01 && t_rel < 0.) {
    char name[80]; sprintf (name, "dump-%06.2f", t);
    dump (file = name);
  }
}

void write_frame (int n) {
  char name[80];
  /* dye: 2400 x 480 px, 0.25 mm/px = 0.025 D, row 0 = +y */
  if (n % nout_every == 0) {
    sprintf (name, "raw/dye-%04d.u16", n);
    FILE * fp = fopen (name, "wb");
    unsigned short * row = malloc (sizeof(unsigned short)*2400*3);
    for (int j = 0; j < 480; j++) {
      double yy = 6. - (j + 0.5)*0.025;
      for (int ii = 0; ii < 2400; ii++) {
	double xx = -30. + (ii + 0.5)*0.025;
	if (xx < XN - 0.3) { row[3*ii] = row[3*ii+1] = row[3*ii+2] = 0; continue; }
	double v0 = interpolate (c0, xx, yy), v1 = interpolate (c1, xx, yy), v2 = interpolate (c2, xx, yy);
	double vv[3] = {v0, v1, v2};
	for (int k = 0; k < 3; k++) {
	  double q = vv[k] == nodata ? 0. : clamp (vv[k], 0., 1.);
	  row[3*ii + k] = (unsigned short) (q*65535. + 0.5);
	}
      }
      fwrite (row, sizeof(unsigned short), 2400*3, fp);
    }
    free (row); fclose (fp);
  }
  /* velocity + vorticity on 1 mm (0.1 D) grid 600 x 120, row 0 = +y */
  scalar om[];
  vorticity (u, om);
  sprintf (name, "raw/uvw-%04d.f32", n);
  FILE * fp = fopen (name, "wb");
  float * buf = malloc (sizeof(float)*600*3);
  for (int j = 0; j < 120; j++) {
    double yy = 6. - (j + 0.5)*0.1;
    for (int ii = 0; ii < 600; ii++) {
      double xx = -30. + (ii + 0.5)*0.1;
      double a1 = interpolate (u.x, xx, yy), a2 = interpolate (u.y, xx, yy), a3 = interpolate (om, xx, yy);
      buf[3*ii] = a1 == nodata ? NAN : a1;
      buf[3*ii+1] = a2 == nodata ? NAN : a2;
      buf[3*ii+2] = a3 == nodata ? NAN : a3;
    }
    fwrite (buf, sizeof(float), 600*3, fp);
  }
  free (buf); fclose (fp);
}

/* v002: velocity history on a 0.5 mm grid (x in [-0.10, 0.30] m, full width) every 1/24 s for Lagrangian dye */

event velhist (t += 1./16.) {
  if (t < TV0 - 1e-9) return 0;
  char name[80]; sprintf (name, "vel/v-%09.4f.f32", t);
  FILE * fp = fopen (name, "wb");
  float * buf = malloc (sizeof(float)*NVX*2);
  for (int j = 0; j < 240; j++) {
    double yy = 6. - (j + 0.5)*0.05;
    for (int ii = 0; ii < NVX; ii++) {
      double xx = VX0 + (ii + 0.5)*0.05;
      double a1 = interpolate (u.x, xx, yy), a2 = interpolate (u.y, xx, yy);
      buf[2*ii] = a1 == nodata ? 0. : a1; buf[2*ii+1] = a2 == nodata ? 0. : a2;
    }
    fwrite (buf, sizeof(float), NVX*2, fp);
  }
  free (buf); fclose (fp);
}

event frames (t += 1./16.) {
  if (t0 < 0. || t < t0 - 1e-6) return 0;
  int n = (int) floor ((t - t0)/DTF + 0.5);
  if (n > 240) return 0;
  write_frame (n);
  fprintf (fev, "frame %d t=%.6f\n", n, t); fflush (fev);
  if (n == 240) {
    fprintf (fev, "done t0=%g t_rel=%g t_front=%g\n", t0, t_rel, t_front);
    fflush (fev); fflush (flog);
    dump (file = "dump-end");   /* v002: allows extending / shifting the window later */
    exit (0);
  }
}

event adapt (i++) {
  if (t_rel >= 0.)
    adapt_wavelet ({cs, u, c0, c1, c2}, (double[]){1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2}, maxlevel, 4);
  else
    adapt_wavelet ({cs, u}, (double[]){1e-3, 1e-2, 1e-2}, maxlevel, 4);
  /* needle-tip source region always at max level (source disk must be resolved) */
  if (t_rel < 0. || t > t_rel - 1.)
    refine (fabs(x - XN) < 0.25 && fabs(y) < 2.7 && level < maxlevel);
}

event end (t = 1e9) {}
