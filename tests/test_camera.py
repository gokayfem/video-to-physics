"""Camera: projection <-> ray-plane round trip, Snell refraction vs a hand-computed case, Blender conversion."""
import math

import numpy as np

from v2p.camera import Camera, blender_camera_params, snell, surface_point


def _cam():
    return Camera.look_at(eye=[0.3, -1.2, 0.9], target=[0, 0, 0], size=(640, 480), fx=700, fy=690, cx=331.5, cy=236.25)


def test_json_round_trip():
    c = _cam()
    c2 = Camera.from_dict(c.to_dict())
    assert np.allclose(c.K, c2.K) and np.allclose(c.R, c2.R) and np.allclose(c.t, c2.t) and c.size == c2.size
    assert np.allclose(c.R @ c.R.T, np.eye(3), atol=1e-12)


def test_project_ray_plane_round_trip():
    c = _cam()
    rng = np.random.default_rng(1)
    P = np.c_[rng.uniform(-0.3, 0.3, (50, 2)), np.full(50, 0.05)]
    uv = c.project(P)
    assert np.allclose(c.ray_plane(uv, 0.05), P, atol=1e-10)
    # principal point maps to the optical axis
    o, d = c.rays(np.array([[c.K[0, 2], c.K[1, 2]]]))
    assert np.allclose(d[0], c.R[2], atol=1e-12)


def test_snell_hand_computed():
    # camera 1 m above a water surface (z = 0), incidence 45 deg, point 0.5 m deep.
    # sin(theta_t) = sin45 / 1.333 = 0.530466; tan(theta_t) = 0.625766; offset under water = 0.312883 m
    n = 1.333
    X = np.array([1.0 + 0.5 * 0.625766, 0.0, -0.5])
    S = surface_point(X, np.array([0.0, 0.0, 1.0]), 0.0, n)
    assert abs(S[0] - 1.0) < 1e-5 and abs(S[1]) < 1e-12 and S[2] == 0.0
    d = np.array([1.0, 0.0, -1.0]) / math.sqrt(2)
    t = snell(d, np.array([0.0, 0.0, 1.0]), 1 / n)
    assert abs(np.linalg.norm(t) - 1) < 1e-12
    assert abs(t[0] - 0.530466) < 1e-5 and abs(t[2] + math.sqrt(1 - 0.530466 ** 2)) < 1e-5


def test_refracted_round_trip():
    c = _cam()
    refract = (1.333, 0.02)
    rng = np.random.default_rng(2)
    P = np.c_[rng.uniform(-0.25, 0.25, (40, 2)), np.full(40, -0.03)]
    uv = c.project(P, refract=refract)
    assert np.allclose(c.ray_plane(uv, -0.03, refract=refract), P, atol=1e-8)
    # refraction makes submerged points appear closer to the nadir than without it
    assert not np.allclose(uv, c.project(P), atol=1.0)


def test_blender_params():
    c = Camera.look_at(eye=[0, 0, 2], target=[0, 0, 0], size=(1920, 1080), fx=1500, cx=960, cy=540)
    p = blender_camera_params(c)
    assert abs(p["lens_mm"] - 1500 * 36 / 1920) < 1e-9 and p["sensor_fit"] == "HORIZONTAL"
    assert abs(p["shift_x"]) < 1e-12 and abs(p["shift_y"]) < 1e-12
    M = np.array(p["matrix_world"])
    assert np.allclose(M[:3, 3], [0, 0, 2])
    assert np.allclose(M[:3, 2], [0, 0, 1], atol=1e-12)     # Blender camera looks along its local -Z
