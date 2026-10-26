/* Force on the cylinder only (embed_force restricted to a box around the cylinder,
   because the channel walls are embedded boundaries too). Copied from embed.h embed_force. */
void cyl_force (scalar p, vector u, face vector mu, double xc, double yc, double rad,
		coord * Fp, coord * Fmu)
{
  coord Fps = {0}, Fmus = {0};
  foreach (reduction(+:Fps) reduction(+:Fmus), nowarning)
    if (cs[] > 0. && cs[] < 1. && fabs(x - xc) < rad && fabs(y - yc) < rad) {
      coord n, b;
      double area = embed_geometry (point, &b, &n);
      area *= pow (Delta, dimension - 1);
      double Fn = area*embed_interpolate (point, p, b);
      foreach_dimension()
	Fps.x += Fn*n.x;
      double mua = 0., fa = 0.;
      foreach_dimension() {
	mua += mu.x[] + mu.x[1];
	fa  += fm.x[] + fm.x[1];
      }
      mua /= fa;
      coord dudn = embed_gradient (point, u, b, n);
      foreach_dimension()
	Fmus.x -= area*mua*(dudn.x*(sq(n.x) + 1.) + dudn.y*n.x*n.y);
    }
  *Fp = Fps; *Fmu = Fmus;
}
