# Credits and Sources

All code in this repository is an original implementation released under the MIT license (see LICENSE): the
`v2p` tools and library, the Blender scripts, the templates and the worked examples. Published results are used as
benchmarks and are cited where they are used.

| Technique, benchmark or material | Source | Used for |
|---|---|---|
| Euler critical load of a clamped–free column, P_cr = π²EI/4L² | L. Euler (1744); Timoshenko & Gere, *Theory of Elastic Stability* | elastica benchmarks |
| Elastica tip deflection via complete elliptic integrals; eccentric tip load kL·tan kL = L/e; heavy column (Greenhill) | Timoshenko & Gere; A. G. Greenhill (1881) | elastica benchmarks |
| Generalised-α time integration | J. Chung, G. M. Hulbert, *J. Appl. Mech.* 60 (1993) | rod dynamics |
| Flow past a cylinder benchmark 2D-2 | M. Schäfer, S. Turek, "Benchmark computations of laminar flow around a cylinder" (1996) | CFD verification |
| Strouhal–Reynolds relation for laminar shedding | C. H. K. Williamson, *Annu. Rev. Fluid Mech.* 28 (1996) | CFD targets |
| Cylinder drag at low Reynolds number | R. D. Henderson (1995); J. Park, K. Kwon, H. Choi (1998) | CFD targets |
| Grid convergence index | P. J. Roache, *Verification and Validation in Computational Science and Engineering* (1998) | convergence checks |
| Basilisk (adaptive Navier–Stokes, embedded boundaries) | S. Popinet, basilisk.fr (GPL) | used in the vortex example; not redistributed |
| FEniCSx / DOLFINx, gmsh | the FEniCS Project (LGPL); C. Geuzaine, J.-F. Remacle (GPL) | independent CFD verifier in the vortex example; not redistributed |
| Blender, Cycles, Mesh Cache modifier, PC2 format | Blender Foundation, Blender 5.2 manual and Python API | rendering and bakes through the public API |
| Image-to-video generation | FLUX 3 Max image-to-video on fal (fal.ai) | the AI reference clips of the examples; clips are linked, not redistributed in the repository |
| Release film format | the sibling releases Video 2 GLSL (gokayfem/shader-skills) and Video 2 Blender Nodes (gokayfem/video-to-blender) | `v2p.py release` |
