"""Video 2 Physics toolkit: measurement, composition, review and run-hygiene helpers.

Plain Python 3.10+ on numpy, opencv-python and Pillow (scipy optional). Every function takes explicit
paths/arrays; nothing in this package knows about a particular clip, scene or phenomenon.
Modules are imported lazily by `scripts/v2p.py`; import the one you need directly, e.g.
`from v2p.camera import Camera`.
"""
__version__ = "0.1.0"
