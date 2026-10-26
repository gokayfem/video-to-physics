"""Role E independent reference solver: FEniCSx (dolfinx 0.11) IPCS, Taylor-Hood P2/P1, gmsh mesh.
Independent of the production (Basilisk) solver; written from the standard incremental pressure-correction scheme
(Goda 1979 / Langtangen-Logg), semi-implicit Adams-Bashforth convection + Crank-Nicolson diffusion.

Cases (nondimensional input, SI-like units as given):
  st2d2 : Schafer-Turek 2D-2, channel 2.2 x 0.41, D=0.1 at (0.2,0.2), parabolic inflow Umax=1.5 (mean 1), nu=1e-3 (Re 100)
  prod  : production channel scaled by D: width 12 D, cylinder on centreline, parabolic inflow mean 1, nu = 1/Re
Usage: mpiexec -n 3 python ipcs_cylinder.py --case st2d2 --hcyl 0.004 --hfar 0.03 --dt 5e-4 --T 8 --out ../out/ref_st2d2_L1.npz
"""
import argparse, json, time, sys
import numpy as np
from mpi4py import MPI
import gmsh, ufl
from petsc4py import PETSc
from dolfinx import fem, geometry
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector, create_matrix


def cvec(L):
    try:
        return create_vector(L)
    except Exception:
        return create_vector(L.function_spaces[0])
from basix.ufl import element

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='st2d2')
ap.add_argument('--hcyl', type=float, default=0.004)
ap.add_argument('--hfar', type=float, default=0.03)
ap.add_argument('--dt', type=float, default=5e-4)
ap.add_argument('--T', type=float, default=8.0)
ap.add_argument('--Re', type=float, default=150.0)
ap.add_argument('--xin', type=float, default=-10.0, help='prod: inlet x (in D) rel. cylinder')
ap.add_argument('--xout', type=float, default=25.0, help='prod: outlet x (in D) rel. cylinder')
ap.add_argument('--W', type=float, default=12.0, help='prod: channel width in D')
ap.add_argument('--kick', type=float, default=0.5, help='cylinder spin amplitude (U units) during t<tkick to trigger shedding')
ap.add_argument('--tkick', type=float, default=-1)
ap.add_argument('--out', required=True)
ap.add_argument('--inflow', default='parabolic')
ap.add_argument('--nocyl', action='store_true', help='empty channel (U_in calibration); u_up5D probe moved to x_cyl')
ap.add_argument('--Uin', type=float, default=1.0, help='prod plug inflow speed (U units)')
ap.add_argument('--drag', type=float, default=0.0, help='linear friction lambda*D/U')
a = ap.parse_args()
comm = MPI.COMM_WORLD
rank = comm.rank

if a.case == 'st2d2':
    L, H, xc, yc, D, nu, Umean = 2.2, 0.41, 0.2, 0.2, 0.1, 1e-3, 1.0
    x0, y0 = 0.0, 0.0
    probe = (xc + 0.3, yc)          # wake probe 3D behind centre
    tkick = 0.5 if a.tkick < 0 else a.tkick
else:
    D, nu, Umean = 1.0, 1.0 / a.Re, 1.0
    x0, y0 = a.xin, -a.W / 2
    L, H = a.xout - a.xin, a.W
    xc, yc = 0.0, 0.0
    probe = (xc + 3.0, yc)
    tkick = 5.0 if a.tkick < 0 else a.tkick
rho = 1.0
Umax = 1.5 * Umean

# ---------------- mesh ----------------
gmsh.initialize(['', '-v', '0'] if True else [])
gmsh.option.setNumber('General.Terminal', 0)
if rank == 0:
    rect = gmsh.model.occ.addRectangle(x0, y0, 0, L, H)
    if not a.nocyl:
        disk = gmsh.model.occ.addDisk(xc, yc, 0, D / 2, D / 2)
        fl = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [vols[0][1]], 1)
    inl, outl, walls, obst = [], [], [], []
    for d, t in gmsh.model.getBoundary(vols, oriented=False):
        com = gmsh.model.occ.getCenterOfMass(1, t)
        if np.isclose(com[0], x0): inl.append(t)
        elif np.isclose(com[0], x0 + L): outl.append(t)
        elif np.isclose(com[1], y0) or np.isclose(com[1], y0 + H): walls.append(t)
        else: obst.append(t)
    for tag, ents in [(2, inl), (3, outl), (4, walls), (5, obst)]:
        gmsh.model.addPhysicalGroup(1, ents, tag)
    if a.nocyl:
        gmsh.model.addPhysicalGroup(0, [], 99) if False else None
        p0 = gmsh.model.occ.addPoint(xc, yc, 0); gmsh.model.occ.synchronize()
    f1 = gmsh.model.mesh.field.add('Distance')
    if a.nocyl: gmsh.model.mesh.field.setNumbers(f1, 'PointsList', [p0])
    else: gmsh.model.mesh.field.setNumbers(f1, 'EdgesList', obst)
    f2 = gmsh.model.mesh.field.add('Threshold')
    gmsh.model.mesh.field.setNumber(f2, 'IField', f1)
    gmsh.model.mesh.field.setNumber(f2, 'LcMin', a.hcyl)
    gmsh.model.mesh.field.setNumber(f2, 'LcMax', a.hfar)
    gmsh.model.mesh.field.setNumber(f2, 'DistMin', 0.5 * D)
    gmsh.model.mesh.field.setNumber(f2, 'DistMax', 4 * D)
    # refined wake box
    f3 = gmsh.model.mesh.field.add('Box')
    gmsh.model.mesh.field.setNumber(f3, 'VIn', min(a.hfar, 4 * a.hcyl * (1 if a.case == 'st2d2' else 1.5)))
    gmsh.model.mesh.field.setNumber(f3, 'VOut', a.hfar)
    gmsh.model.mesh.field.setNumber(f3, 'XMin', xc - D); gmsh.model.mesh.field.setNumber(f3, 'XMax', xc + (1.2 if a.case == 'st2d2' else 12) * (1 if a.case != 'st2d2' else 1))
    gmsh.model.mesh.field.setNumber(f3, 'YMin', yc - 2 * D); gmsh.model.mesh.field.setNumber(f3, 'YMax', yc + 2 * D)
    fm = gmsh.model.mesh.field.add('Min'); gmsh.model.mesh.field.setNumbers(fm, 'FieldsList', [f2, f3])
    gmsh.model.mesh.field.setAsBackgroundMesh(fm)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.Algorithm', 6)
    gmsh.model.mesh.generate(2)
try:
    from dolfinx.io import gmsh as gmshio
except ImportError:
    from dolfinx.io import gmshio
md = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
if isinstance(md, tuple):
    msh, ct, ft = md[0], md[1], md[2]
else:
    msh, ft = md.mesh, md.facet_tags
gmsh.finalize()
ncell = comm.allreduce(msh.topology.index_map(2).size_local, op=MPI.SUM)

# ---------------- spaces / bcs ----------------
V = fem.functionspace(msh, element('Lagrange', msh.basix_cell(), 2, shape=(2,)))
Q = fem.functionspace(msh, element('Lagrange', msh.basix_cell(), 1))
fdim = 1
ndofs = comm.allreduce(V.dofmap.index_map.size_local, op=MPI.SUM) * 2


class Inflow:
    def __call__(self, x):
        s = (x[1] - y0) / H
        v = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        v[0] = a.Uin if a.inflow == 'plug' else 4 * Umax * s * (1 - s)
        return v


class Spin:
    def __init__(self): self.w = 0.0
    def __call__(self, x):
        v = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        v[0] = -self.w * (x[1] - yc) / (D / 2)
        v[1] = self.w * (x[0] - xc) / (D / 2)
        return v


u_in = fem.Function(V); u_in.interpolate(Inflow())
spin = Spin(); u_cyl = fem.Function(V); u_cyl.interpolate(spin)
u_zero = fem.Function(V); u_zero.x.array[:] = 0
bcu = [fem.dirichletbc(u_in, fem.locate_dofs_topological(V, fdim, ft.find(2))),
       fem.dirichletbc(u_zero, fem.locate_dofs_topological(V, fdim, ft.find(4))),
       fem.dirichletbc(u_cyl, fem.locate_dofs_topological(V, fdim, ft.find(5)))]
bcp = [fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Q, fdim, ft.find(3)), Q)]

# ---------------- forms ----------------
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
u_ = fem.Function(V); u_s = fem.Function(V); u_n = fem.Function(V); u_n1 = fem.Function(V)
p_ = fem.Function(Q); phi = fem.Function(Q)
k = fem.Constant(msh, PETSc.ScalarType(a.dt))
mu = fem.Constant(msh, PETSc.ScalarType(nu * rho))
rr = fem.Constant(msh, PETSc.ScalarType(rho))
F1 = rr / k * ufl.dot(u - u_n, v) * ufl.dx
F1 += ufl.inner(ufl.dot(1.5 * u_n - 0.5 * u_n1, 0.5 * ufl.nabla_grad(u + u_n)), v) * ufl.dx
if a.drag > 0:
    lam = fem.Constant(msh, PETSc.ScalarType(a.drag))
    F1 += rr * lam * ufl.dot(0.5 * (u + u_n), v) * ufl.dx
F1 += 0.5 * mu * ufl.inner(ufl.grad(u + u_n), ufl.grad(v)) * ufl.dx - ufl.dot(p_, ufl.div(v)) * ufl.dx
a1 = fem.form(ufl.lhs(F1)); L1 = fem.form(ufl.rhs(F1))
A1 = create_matrix(a1)
b1 = cvec(L1)
a2 = fem.form(ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx)
L2 = fem.form(-rr / k * ufl.div(u_s) * q * ufl.dx)
A2 = assemble_matrix(a2, bcs=bcp); A2.assemble(); b2 = cvec(L2)
a3 = fem.form(rr * ufl.dot(u, v) * ufl.dx)
L3 = fem.form(rr * ufl.dot(u_s, v) * ufl.dx - k * ufl.dot(ufl.nabla_grad(phi), v) * ufl.dx)
A3 = assemble_matrix(a3); A3.assemble(); b3 = cvec(L3)


def mk(A, kind):
    s = PETSc.KSP().create(comm); s.setOperators(A)
    if kind == 'lu':
        s.setType(PETSc.KSP.Type.PREONLY); pc = s.getPC(); pc.setType(PETSc.PC.Type.LU)
        try: pc.setFactorSolverType('mumps')
        except Exception: pass
    elif kind == 'gmres':
        s.setType(PETSc.KSP.Type.BCGS); s.getPC().setType(PETSc.PC.Type.JACOBI); s.setTolerances(rtol=1e-10)
    else:
        s.setType(PETSc.KSP.Type.CG); pc = s.getPC(); pc.setType(PETSc.PC.Type.GAMG); s.setTolerances(rtol=1e-10)
    return s


s1 = mk(A1, 'gmres'); s2 = mk(A2, 'lu'); s3 = mk(A3, 'cg')

nrm = -ufl.FacetNormal(msh)   # pointing out of obstacle into fluid
dObs = ufl.Measure('ds', domain=msh, subdomain_data=ft, subdomain_id=5)
u_t = ufl.inner(ufl.as_vector((nrm[1], -nrm[0])), u_)
coef = 2 / (rho * Umean ** 2 * D)
drag = fem.form(coef * (mu * ufl.inner(ufl.grad(u_t), nrm) * nrm[1] - p_ * nrm[0]) * dObs)
lift = fem.form(-coef * (mu * ufl.inner(ufl.grad(u_t), nrm) * nrm[0] + p_ * nrm[1]) * dObs)
divL2 = fem.form(ufl.div(u_) ** 2 * ufl.dx)
ke = fem.form(0.5 * ufl.dot(u_, u_) * ufl.dx)

# probe
tree = geometry.bb_tree(msh, msh.topology.dim)
pt = np.array([[probe[0], probe[1], 0.0]])
cand = geometry.compute_collisions_points(tree, pt)
coll = geometry.compute_colliding_cells(msh, cand, pt)
pcell = coll.links(0)
has_probe = len(pcell) > 0


pt2 = np.array([[xc if a.nocyl else xc - 5 * D, yc, 0.0]])
pcell2 = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pt2), pt2).links(0)


def probe_u_up():
    hp = len(pcell2) > 0
    val = u_.eval(pt2, pcell2[:1])[0] if hp else 0.0
    for h_, vv in comm.allgather((hp, float(np.real(val)))):
        if h_: return vv
    return np.nan


def probe_v():
    val = u_.eval(pt, pcell[:1])[1] if has_probe else 0.0
    vals = comm.allgather((has_probe, float(np.real(val))))
    for hp, vv in vals:
        if hp: return vv
    return np.nan


nsteps = int(round(a.T / a.dt))
ts, cds, cls, pvs, pus = [], [], [], [], []
t = 0.0
t0 = time.time()
if rank == 0:
    print(f'case={a.case} cells={ncell} vel_dofs={ndofs} steps={nsteps}', flush=True)
for i in range(nsteps):
    t += a.dt
    w = a.kick * np.sin(np.pi * t / tkick) if t < tkick else 0.0
    if w != spin.w or i == 0:
        spin.w = w; u_cyl.interpolate(spin)
    # step 1
    A1.zeroEntries(); assemble_matrix(A1, a1, bcs=bcu); A1.assemble()
    with b1.localForm() as l: l.set(0)
    assemble_vector(b1, L1); apply_lifting(b1, [a1], [bcu]); b1.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    s1.solve(b1, u_s.x.petsc_vec); u_s.x.scatter_forward()
    # step 2
    with b2.localForm() as l: l.set(0)
    assemble_vector(b2, L2); apply_lifting(b2, [a2], [bcp]); b2.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    s2.solve(b2, phi.x.petsc_vec); phi.x.scatter_forward()
    p_.x.petsc_vec.axpy(1, phi.x.petsc_vec); p_.x.scatter_forward()
    # step 3
    with b3.localForm() as l: l.set(0)
    assemble_vector(b3, L3); b3.ghostUpdate(PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
    s3.solve(b3, u_.x.petsc_vec); u_.x.scatter_forward()
    with u_.x.petsc_vec.localForm() as lu_, u_n.x.petsc_vec.localForm() as lun, u_n1.x.petsc_vec.localForm() as lun1:
        lun.copy(lun1); lu_.copy(lun)
    cd = comm.allreduce(fem.assemble_scalar(drag), op=MPI.SUM)
    cl = comm.allreduce(fem.assemble_scalar(lift), op=MPI.SUM)
    ts.append(t); cds.append(cd); cls.append(cl); pvs.append(probe_v()); pus.append(probe_u_up())
    if rank == 0 and (i % 500 == 0 or i == nsteps - 1):
        dv = np.sqrt(comm.allreduce(fem.assemble_scalar(divL2), op=MPI.SUM)) if False else np.nan
        el = time.time() - t0
        print(f't={t:.3f} Cd={cd:.4f} Cl={cl:+.4f} probe_v={pvs[-1]:+.4f} wall={el:.0f}s eta={el/(i+1)*(nsteps-i-1):.0f}s', flush=True)
        np.savez(a.out, t=np.array(ts), Cd=np.array(cds), Cl=np.array(cls), pv=np.array(pvs), u_up5D=np.array(pus),
                 meta=json.dumps(dict(vars(a), cells=ncell, vel_dofs=ndofs, D=D, Umean=Umean, nu=nu, probe=probe, done=False)))
divn = np.sqrt(comm.allreduce(fem.assemble_scalar(divL2), op=MPI.SUM))
if rank == 0:
    np.savez(a.out, t=np.array(ts), Cd=np.array(cds), Cl=np.array(cls), pv=np.array(pvs), u_up5D=np.array(pus),
             meta=json.dumps(dict(vars(a), cells=ncell, vel_dofs=ndofs, D=D, Umean=Umean, nu=nu, probe=probe, done=True,
                                  div_L2_final=divn, wall_s=time.time() - t0)))
    print('done', flush=True)
