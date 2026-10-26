/* Schäfer & Turek (1996) 2D-2 benchmark: Re = 100, channel 2.2 x 0.41, D = 0.1 at (0.2,0.2),
   parabolic inflow Um = 1.5 (mean 1), nu = 1e-3. Usage: ./bench LEVEL TEND */
#include "embed.h"
#include "navier-stokes/centered.h"
#include "cylforce.h"

int maxlevel = 10;
double tend = 10.;
const double H = 0.41, Dc = 0.1, Um = 1.5, nu0 = 1e-3;
face vector muv[];
FILE * flog;

int main (int argc, char * argv[])
{
  if (argc > 1) maxlevel = atoi (argv[1]);
  if (argc > 2) tend = atof (argv[2]);
  L0 = 2.2;
  origin (0., -0.9037);
  N = 64;
  mu = muv;
  TOLERANCE = 1e-4;
  char name[80]; sprintf (name, "forces-%d.dat", maxlevel);
  flog = fopen (name, "w");
  run();
}

event properties (i++) {
  foreach_face()
    muv.x[] = fm.x[]*nu0;
}

u.n[left]  = dirichlet (y > 0. && y < H ? 4.*Um*y*(H - y)/sq(H) : 0.);
u.t[left]  = dirichlet (0.);
p[left]    = neumann (0.);
pf[left]   = neumann (0.);
u.n[right] = neumann (0.);
p[right]   = dirichlet (0.);
pf[right]  = dirichlet (0.);
u.n[embed] = dirichlet (0.);
u.t[embed] = dirichlet (0.);

event init (t = 0) {
  refine (y > -0.02 && y < H + 0.02 && level < maxlevel - 2);
  refine (sq(x - 0.2) + sq(y - 0.2) < sq(0.08) && level < maxlevel);
  solid (cs, fs, min (min (y, H - y), sqrt (sq(x - 0.2) + sq(y - 0.2)) - Dc/2.));
  foreach()
    u.x[] = cs[] ? 4.*Um*y*(H - y)/sq(H) : 0.;
}

event logfile (i++) {
  coord Fp, Fmu;
  cyl_force (p, u, mu, 0.2, 0.2, 0.08, &Fp, &Fmu);
  double c = 2./(sq(1.)*Dc);
  fprintf (flog, "%.6f %.6f %.6f %d %d %ld\n", t, c*(Fp.x + Fmu.x), c*(Fp.y + Fmu.y),
	   mgp.i, mgu.i, grid->tn);
  fflush (flog);
}

event stop (t = tend) {
  fprintf (stderr, "done t=%g i=%d\n", t, i);
}

event adapt (i++) {
  adapt_wavelet ({cs, u}, (double[]){1e-3, 1e-2, 1e-2}, maxlevel, 5);
}
