# pyclipper

This project was originally a set of python bindings for Angus Johnson's excellent [ClipperLib](http://www.angusj.com/delphi/clipper.php) library. However, with time, these bindings became mostly sidelined, and now they are vestigial. Instead, this project has specialized in reading toolpaths and contours from stdin or files. These paths are raw ClipperLib::Paths, with some metadata attached. It can plot these paths in 2D and in 3D. It contains miscellaneous related functionality.

## Dependencies

`CMakeLists.txt` and `setup.py` assume that [a patched version of ClipperLib](https://github.com/jdfr/clipper) is located in ../clipper, but it can be easily adapted to use any reasonably current version of ClipperLib.

From the python ecosystem, pyclipper has the following dependencies:

* Python 2.7.10
* Cython 0.22
* Numpy 1.8.2
* Matplotlib 1.4.2 (optional, for 2D plotting)
* Mayavi 4.3.1 (optional, for 3D plotting)

Please note that the project will probably compile with other versions of the dependencies, the numbers are just the versions used in development. It has been tested only on CPython +2.7 and +3.4, both in Linux and Windows.

## How to compile

After the dependencies have been installed, you can proceed to compilation. To compile the cython bindings, do the following:

```bash
python setup.py build_ext --inplace
```

Alternatively, you can use the provided cmake script to copy the source files to another location and build the Cython extension there. The cmake script is also aware of the complexities of building if you are using a portable version of WinPython, for which the build process is not as straightforward as it is in other distributions.
