# pyslic3r

pyslic3r is a testbed for an experimental python slicing library using slic3r's core C++ codebase.

Because Slic3r is designed as a full-fledged slicing solution, its library has a relatively involved data model, with many C++ classes representing many different slicing concepts and abstractions. In its current form, pyslic3r does not attempt to expose all of slic3r's architecture, but just the most basic bits of it, in order to slice STL files. From a technical point of view, we are just wrapping two kinds of objects:

* `SlicerMesh` wraps `Slic3r::TriangleMesh`

* `SlicedModel` wraps `std::vector<ExPolygons>`, which is the result of `Slic3r::TriangleMeshSlicer::slice`

Some operations on these objects are also wrapped.

## Rationale

slic3r is one of the best open source slicers out there, and has a big, high-quality codebase. Rather than adapt an already existing clipping library to do slicing in python, pyslic3r builds upon alexrj's excellent work in the C++ core of slic3r.

# Compiling

This code has been tested only on debian sid, but it should also work in other distros. To compile on windows, cygwin is needed.

This code has as dependency a (minimally) patched version of Slic3r, which is cloned in deps/Slic3r/Slic3r by cmake.

To compile both Slic3r's C++ core and the cython bindings, do the following:

```bash
cd cmakebuild
cmake ..
make
```

## Porting to other systems

Unfortunately, Slic3r's C++ core has not been coded with Windows portability in mind (it does use the LP64 data model, and Visual Studio complains on many std calls, among other issues), because Perl compiles it in CygWin.

Clipper, on the other hand, is fully portable.

## Notes

pyslic3r uses only a part of Slic3r's C++ codebase, so some parts of the compiled C++ library are not used.

# License

This library is licensed under the AGPLv3.0.


