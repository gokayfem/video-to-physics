# Clip 02 coordinates (lead, kickoff)
- Origin A = the spring's fixed upper attachment point (where the top hook bears on the crossbar).
- Z up (gravity -Z_world), X horizontal to image-right in the plane containing the camera's view of the
  frame front, Y horizontal away from the camera side (right-handed). Final axes refined by camera calibration
  (record the rotation if X is redefined along the crossbar).
- Generalized coordinates expected: 3D position of the weight's attachment (or CoM) relative to A, plus
  weight orientation if it visibly rocks/rotates; spring axis = A -> lower hook.
- Prefix: VSP_ (objects VSP_*, scene camera VSP_Cam_Reference).
