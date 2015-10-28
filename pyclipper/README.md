This directory contains the sources for pyclipper:

* `__init__.py`: so this is a package

* `Clipper_defs.pxd`: declaring `ClipperLib` C++ types for use in cython

* `Clipper.*`: this is the code to wrap the `ClipperLib` library. The main data type is `ClipperPaths`, a wrapper for `ClipperLib::Paths`.

* `plot2d.py`: utilities to display `ClipperPaths` and many other possible objects in `matplotlib`.

* `plot3d.py`: utilities to display sets of slices in `mayavi2`.

* `plot.py`: import `plo2d` and `plot3d` together

* `all.py`: import everything from `pyclipper`