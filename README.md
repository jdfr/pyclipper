# pyslic3r

pyslic3r is a testbed for an experimental python slicing library using slic3r's core C++ codebase.

Because Slic3r is designed as a full-fledged slicing solution, its library has a relatively involved data model, with many C++ classes representing many different slicing concepts and abstractions. In its current form, pyslic3r does not attempt to expose all of slic3r's architecture, but just the most basic bits of it, in order to slice STL files. From a technical point of view, we are just wrapping two kinds of objects:

* `SlicerMesh` wraps `Slic3r::TriangleMesh`

* `SlicedModel` wraps `std::vector<ExPolygons>`, which is the result of `Slic3r::TriangleMeshSlicer::slice`

Some operations on these objects are also wrapped.

## Rationale

slic3r is one of the best open source slicers out there, and has a big, high-quality codebase. Rather than adapt an already existing clipping library to do slicing in python, pyslic3r builds upon alexrj's excellent work in the C++ core of slic3r.

# Compiling

This code has been tested only on debian sid, but it should also work in other distros. To compile on windows, only minor changes should be needed.

Slic3r has to be checked out to `./deps/Slic3r/Slic3r`. To compile slic3r's C++ library, go to `./deps/Slic3r/Slic3r-build` and do:

```python
x = [a+1 for a in xrange(10)]
for q in x:
  print x+1
```


```bash
cmake ..
make
```

Then, go to `./pyslic3r` and do:

```bash
python setup.py build_ext --inplace
```

## Porting to other systems

The use of cmake should make it rather easy to compile slic3r's C++ library in Windows: just run the cmake GUI to generate a Visual Studio project to compile the library. the CMakeLists.txt is configured to generate a static library because it is easier to use in development in that way. However, the proper way to do it is probably to generate a shared library. 

Currently, ./pyslic3r/setup.py is hardcoded to link the static slic3r library generated in my debian system into the cython bindings. It should probably work out of the box in other distros. Porting to windows should be easy.

## Notes

slic3r's C++ codebase is compiled wholesale into the library, although I am using only a tiny bit of it.

