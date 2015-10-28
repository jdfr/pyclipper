# pyclipper

pyclipper is an experimental python binding for Angus Johnson's excellent [ClipperLib](http://www.angusj.com/delphi/clipper.php) library.

## Rationale

Computationally expensive tasks such as clipping and offseting have to be done in a low level language. Numpy, numba and scipy.weave can help, but many geometrical algorithms cannot be vectorized easily, and you just have to use a language closer to the metal, such as C. You should use each tool for the task it does the best: C/C++ for number-crunching, Python for glue logic. Solutions written in pure python are slow, probably even when executed with Pypy.

## Current state

`ClipperXXX` objects encapsulate ClipperLib objects: `ClipperPaths` and `ClipperPolyTree` wrap `ClipperLib:Paths` and `ClipperLib::PolyTree`, respectively, which are the main data types of ClipperLib. `ClipperClip` and `ClipperOffset` encapsulate the clipping and offseting engines, respectively.

Most operations on these objects are also wrapped. Additionally, it is possible to show `ClipperPaths` using matplotlib or mayavi. The contours in these objects are exposed using custom accesors: the contours are casted as numpy matrices representing lists of points.

## Dependencies

`CMakeLists.txt` and `setup.py` assume that [a patched version of ClipperLib](https://github.com/jdfr/clipper) is located in ../clipper, but it can be easily adapted to use any reasonably current version of ClipperLib.

From the python ecosystem, pyclipper has the following dependencies:

* Python 2.7.10
* Cython 0.22
* Numpy 1.8.2
* Matplotlib 1.4.2 (optional, for 2D plotting)
* Mayavi 4.3.1 (optional, for 3D plotting)

Please note that the project will probably compile with other versions of the dependencies, the numbers are just the versions used in development.

## How to compile

After the dependencies have been installed, you can proceed to compilation. To compile the cython bindings, do the following:

```bash
mkdir cmakebuild
cd cmakebuild
cmake ..
make
```

alternatively:

```bash
python setup.py build_ext --inplace
```

