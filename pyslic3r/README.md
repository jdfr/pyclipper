This directory contains the sources for pyslic3r:

* `__init__.py`: so this is a package

* `slic3r_defs.pxd`: declaring `Slic3r` C++ types for use in cython

* `Clipper_defs.pxd`: declaring `ClipperLib` C++ types for use in cython

* `Clipper.*`: this is the code to wrap the `ClipperLib` library. The main data type is `ClipperPaths`, a wrapper for `ClipperLib::Paths`.

* `TriangleMesh.*`: this is the code to wrap `Slic3r::TriangleMesh`, the data type that represents a 3d mesh from a STL file

* `SlicedModel.*`: this is the code to wrap `std::vector<Slic3r::Expolygons>`, the main data type we use from Slic3r to represent slices. The wrapper is `SlicedModel`

* `slicing.py`: import TriangleMesh and SlicedModel together

* `plot2d.py`: utilities to display `ClipperPaths`, `SlicedModel` and many other possible objects in `matplotlib`.

* `plot3d.py`: utilities to display objects with an interface similar to `SlicedModel` in `mayavi2`.

* `plot.py`: import `plo2d` and `plot3d` together

* `all.py`: import everything from `pyslic3r`