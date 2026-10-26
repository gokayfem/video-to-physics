"""Create or update a Blender camera from a camera json (the Video 2 Physics data contract).

usage: blender -b [SCENE.blend] --factory-startup --python camera_from_json.py -- --camera CAM.json
           [--name Cam_Reference] [--per-frame CAMS.json] [--save OUT.blend] [--set-active]
CAM.json:   {"image_size":[W,H], "intrinsics":{fx_px,fy_px,cx_px,cy_px}, "extrinsics":{R_world_to_cam,t}}
CAMS.json:  optional per-frame cameras {"frames": {"0": CAM, "1": CAM, ...}} keyed by frame number; each frame gets
            keyframes for location, rotation, lens and shift (constant interpolation, no smoothing here).
Conversion (pure Python, unit-tested): v2p.camera.blender_camera_params.
"""
import argparse
import json
import os
import sys

import bpy
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2p.camera import Camera, blender_camera_params  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(prog="camera_from_json.py")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--name", default="Cam_Reference")
    ap.add_argument("--per-frame")
    ap.add_argument("--save")
    ap.add_argument("--set-active", action="store_true", default=True)
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def apply(obj, p, scene, frame=None):
    cam = obj.data
    cam.sensor_fit = p["sensor_fit"]
    cam.sensor_width = cam.sensor_height = p["sensor_mm"]
    cam.lens, cam.shift_x, cam.shift_y = p["lens_mm"], p["shift_x"], p["shift_y"]
    obj.matrix_world = Matrix(p["matrix_world"])
    scene.render.resolution_x, scene.render.resolution_y = p["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = p["pixel_aspect_x"], p["pixel_aspect_y"]
    if frame is not None:
        bpy.context.view_layer.update()
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_euler", frame=frame)
        for path in ("lens", "shift_x", "shift_y"):
            cam.keyframe_insert(path, frame=frame)


def main():
    a = parse()
    scene = bpy.context.scene
    obj = bpy.data.objects.get(a.name)
    if obj is None:
        obj = bpy.data.objects.new(a.name, bpy.data.cameras.new(a.name))
        scene.collection.objects.link(obj)
    obj.rotation_mode = "XYZ"
    apply(obj, blender_camera_params(Camera.from_json(a.camera)), scene)
    if a.per_frame:
        frames = json.load(open(a.per_frame))["frames"]
        for k in sorted(frames, key=int):
            apply(obj, blender_camera_params(Camera.from_dict(frames[k])), scene, int(k))
        for data in (obj.animation_data, obj.data.animation_data):
            if data and data.action:
                for fc in getattr(data.action, "fcurves", []):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"
    if a.set_active:
        scene.camera = obj
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
    print(f"camera {obj.name}: lens {obj.data.lens:.3f} mm, shift ({obj.data.shift_x:.4f}, {obj.data.shift_y:.4f})")


main()
