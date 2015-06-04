# pyslic3r

pyslic3r is an experimental python slicing library using [Slic3r](https://github.com/alexrj/Slic3r/)'s core [C++ codebase](https://github.com/alexrj/Slic3r/tree/master/xs/src).

## Intended use

With conventional slicers, the workflow is roughly as this:

1. 3D object input
2. slicing
3. add supporting material
4. infilling
5. motion planning
6. gcode generation

Fantastic for end users. But what if you want to test some new idea on how to generate or modify the slices? Most slicers are configurable to some extent, but I have not seen anyone that is fully scriptable. This is the intended use of pyslic3r:

1. 3D object input
2. slicing
3. arbitrary modification of the slices

In an ideal world, pyslic3r would expose the entire workflow of Slic3r in a fully scriptable way. Unfortunately, substantial portions of Slic3r's logic are in Perl, so using them from pyslic3r is not currently practical. As a workaround, a [patched version of slic3r](https://github.com/jdfr/Slic3r/tree/pychanges) is provided, to enable the following workflow:

1. use pyslic3r. Save the slices to a file in an adhoc binary format.
2. load the file in the patched version of slic3r, to automatically do all subsequent operations (support material, infilling, planning, gcode generation). If you have generated a file witht the slices named sliced.paths, you can use the patched version of slic3r like this:

```bash
perl path/to/slic3r/slic3r.pl --import-paths=path/to/data/sliced.paths x
```

The above command should generate the file `path/to/data/sliced.gcode`. See `examples/ex05.py` for an example on how to generate the files that this patched version of Slic3r reads as input. 

To aid development, two visualization kits are included to show and inspect slices and stacks of slices, one uses matplotlib (2D visualization) and the other mayavi (3D visualization).

## Rationale

Slic3r is one of the best open source slicers out there, and has a big, high-quality codebase. Rather than adapt an already existing clipping library to do slicing in python, pyslic3r builds upon the excellent codebase in the C++ core of slic3r.

Of course, [Skeinforge](http://reprap.org/wiki/Skeinforge) (the grandfather of open source slicers) is already in python. However, it is very slow. Even when executed with Pypy, it is still slow. On top of that, development has almost come to a halt. I use python because it is incredibly easy to do high-level programming, and matplotlib and mayavi are great to assist the debugging. However, computationally expensive tasks such as slicing, clipping and offseting have to be done in a low level language. Numpy, numba and scipy.weave can help, but many geometrical algorithms cannot be vectorized easily, and you just have to use a language closer to the metal, such as C. You should use each tool for the task it does the best: C/C++ for number-crunching, Python for glue logic. 

## Current state

Because Slic3r is a full-fledged slicing solution, its library has a relatively complex data model, with many C++ classes representing many different slicing concepts and abstractions. In its current form, pyslic3r does not attempt to expose all of this architecture, but just the most basic bits of it, just enough to get to slice STL files. From a technical point of view, we are currently wrapping just three kinds of objects:

* `TriangleMesh` wraps `Slic3r::TriangleMesh`

* `SlicedModel` wraps `std::vector<Slic3r::ExPolygons>`, which is the result of `Slic3r::TriangleMeshSlicer::slice()`.

* `ClipperPaths` wraps `ClipperLib::Paths`, which is the main data type of `ClipperLib`, the clipping & offesting library used internally by Slic3r.

Some operations on these objects are also wrapped. Additionally, it is possible to show 2D and 3D views of `SlicedModel`, and 2D views of `ClipperPaths`. The contours in these objects are exposed using custom accesors as well as slicing notation, casting the contours as numpy matrices representing lists of points.

`ClipperXXX` objects encapsulate ClipperLib objects: `ClipperPaths` and `ClipperPolyTree` wrap `ClipperLib:Paths` and `ClipperLib::PolyTree`, respectively, which are the main data types of ClipperLib. `ClipperClip` and `ClipperOffset` encapsulate the clipping and offseting engines, respectively.

pyslic3r's main data type is `SlicedModel`. Since it can be a bit rigid for a fully pythonic manipulation of the data, it can be converted from/to an arrangement of python objects which are more easily modifiable: `SliceCollection`, `Layer`, and `Expolygon`. `Slic3r::Polygon` is exposed as an bidimensional numpy array of 64-bit integers, so that the each row represents the coordinates of a point.

`ClipperPaths`, `ClipperPolyTree`, `ClipperClip` and `ClipperOffset` encapsulate most of the functionality of ClipperLib. The interface is currently very low-level. Slices from `SlicedModel` can be converted from/to `ClipperPaths` objects.

**Please note**: the use of ClipperLib is an internal implementation detail of Slic3r, and may change in future releases, as expressly stated by its developers.

## Examples

A few examples on how to use the bindings are provided in the directory `examples`:

* `ex01.py` shows how to slice a model, and how to access the results stored in `SlicedModel` wrappers.
* `ex02.py` shows how to use `ClipperPaths` wrappers. These are somewhat redundant with `SlicedModel`, but useful to work in Windows (see section on [compiling on windows](#porting-to-windows)).
* `ex03.py` shows how to do 2D plots of sliced objects.
* `ex04.py` shows how to do 3D plots of sliced objects.
* `ex05.py` shows how to do simple slice processing.

# Compiling on Linux

This code has been tested only on debian jessie, but it should also work in other distros.

## Dependencies

This code depends on a (minimally) [patched version of Slic3r](https://github.com/jdfr/Slic3r/tree/pychanges), which is cloned from github into `deps/Slic3r/Slic3r` by the cmake script. Angus Johnson's ClipperLib is also an implicit dependency, brought along with Slic3r.

From the python ecosystem, pyslic3r has the following dependencies:

* Python 2.7.10
* Cython 0.22
* Numpy 1.8.2
* Matplotlib 1.4.2 (optional, for 2D plotting)
* Mayavi 4.3.1 (optional, for 3D plotting)

Please note that the project will probably compile with other versions of the dependencies, the numbers are just the versions used in development.

## How to compile

After the dependencies have been installed, you can proceed to compilation. To compile both Slic3r's C++ core and the cython bindings, do the following:

```bash
mkdir cmakebuild
cd cmakebuild
cmake ..
make
```

If you want to reload Slic3r's source, the best way is to remove `deps/Slic3r/Slic3r` and the contents of `deps/Slic3r/cmakebuild`, and to run `cmake .. && make` again in `deps/Slic3r/cmakebuild`.

If you want to clean the cmake configuration directories to force full reconfiguration, you can do it with `cd cmakebuild && make cmakeclean` or `python clearcmake.py`.

If you change the cython sources, they can be rebuilt either with `cd cmakebuild && make` or `python setup.py build_ext --inplace`. However, if you change the build options in `setup.py` or `build.py`, no recompilinig will be done. To force recompiling, you will have to remove the built files. Depending on the changes, you may want to remove just the `*.so` files or also the `*.cpp` ones. You can do the later with `cd cmakebuild && make pyclean` or `python clearcython.py`.

**IMPORTANT**: please note that all of this is only for compiling pyslic3r. If you also want the patched version of slic3r to transform the results of pyslic3r into gcode, you also have to separately compile the copy of Slic3r in `deps/Slic3r/Slic3r`, as described in [Slic3r's downloads page](http://slic3r.org/download). **ALSO**: this patched version has been tested only in Linux, and only from the command line

## Python 2 vs 3

pyslic3r has been developed on python 2.7.10. Support for python 3.x is planned but is low priority for now.

## Porting to Windows

The most practical way to compile a full version of pyslic3r in Windows is probably to use Cygwin. There are several hurdles to compile it in Visual Studio:

* Apparently, Slic3r for windows is compiled in a POSIX environment, because Slic3r implicitly uses the LP64 data model, so it should be patched to be data-model agnostic. Probably, it should also be patched for Visual Studio compilation quirks, if there are any. Clipper, on the other hand, is fully portable at the moment.

* Official releases of python 2.x for windows are compiled with Visual Studio 2008, whose supported flavor of C++ is ancient; Slic3r would probably require extensive patches to compile there. You can always see if it works nicely in python 3 (the official releases are compiled with VS 2010), or even go for recompiling all python 2.7 dependencies from scratch in a newer version of Visual Studio, if that is your cup of tea.

* The project has several Cython modules, so it compiles slic3r and ClipperLib as shared libraries to be used by these modules. To be compiled as a shared libraries in Visual Studio, slic3r's and ClipperLib's code should be patched to explicitly export the desired functions and classes. Alternatively, the Cython modules can be refactored into one giant module, statically linking Slic3r to it.

Currently, only the Clipper module can be compiled in windows, so slicing is not supported. Because of this, basic support has been added to work with `ClipperPaths` in windows, if necessary.

# Further development

pyslic3r uses only a part of Slic3r's C++ codebase. Obvious candidates to extend the bindings are:

* The motion planner
* The medial axis algorithm
* Extrusion entities
* The bridge detector
* Functionality in Geometry.cpp and maybe ClipperUtils.cpp

Also, the patch to enable slic3r to be fed the slices generated by pyslic3r is a quick and dirty hack, it will not (and should not, either) be merged into mainstream Slic3r. A more proper long-term solution would be to read the slices from a SVG file with the same format as currently used by Slic3r when the option `--export-svg` is used.

# License

This library is licensed under the AGPLv3.0.

# Origins of the project

I am working for a university in collaboration with a company to write specialized software to process contours for layer manufacturing hardware. The company has set up a slicing pipeline around a proprietary CAD/CAM package. However, the university desired to be able to adapt the software to other layer manufacturing systems, so writing my software in the proprietary framework seemed inadecuate. On top of that, for several reasons I had to start developing my software without direct access to the company's pipeline, so I naturally reached for Slic3r to fill in the bits and concentrate in developing my software, not reinventing the wheel.