This directory contains the sources for pyslic3r:

* `slic3r_defs.pxd`: declaring Slic3r C++ types to use in cython

* `_TriangleMesh.*`: this is the code to wrap Slic3r::TriangleMesh, the data type that represents a STL file

* `_SlicedModel.*`: this is the code to wrap std::vector<Slic3r::Expolygons>, the main data type we use from Slic3r to represent slices. The wrapper is SlicedModel

* `__init__.py`: imports both _TriangleMesh and _SlicedModel. The rest of files have to be imported separately

* `plot.py`: utilities to display SlicedModel in mayavi and matplotlib (not really OK with the later, since matplotlib does not have a true zbuffer).

