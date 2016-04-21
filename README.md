# pyclipper

This version of pyclipper is a very incomplete python binding for Angus Johnson's excellent [ClipperLib](http://www.angusj.com/delphi/clipper.php) library. It only wraps the interface types, not the core functionality of the library. It is prepared to read from a file (or from stdin) a binary stream representing a series of ClipperLib::Paths with several different attributes at different Z levels. It can plot these paths in 2D and in 3D. It contains miscellaneous related functionality.

## Dependencies

`CMakeLists.txt` and `setup.py` assume that [a patched version of ClipperLib](https://github.com/jdfr/clipper) is located in ../clipper, but it can be easily adapted to use any reasonably current version of ClipperLib.

From the python ecosystem, pyclipper has the following dependencies:

* Python 2.7.10
* Cython 0.22
* Numpy 1.8.2
* Matplotlib 1.4.2 (optional, for 2D plotting)
* Mayavi 4.3.1 (optional, for 3D plotting)

Please note that the project will probably compile with other versions of the dependencies, the numbers are just the versions used in development. It has been tested only on CPython.

## How to compile

After the dependencies have been installed, you can proceed to compilation. To compile the cython bindings, do the following:

```bash
python setup.py build_ext --inplace
```

Alternatively, you can use the provided cmake script to copy the source files to another location and build the Cython extension there. The cmake script is also aware of the complexities of building if you are using a portable version of WinPython, for which the build process is not as straightforward as it is in other distributions.