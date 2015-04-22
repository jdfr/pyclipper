# pyslic3r

pyslic3r is a testbed for a python library exposing slic3r's core C++ codebase.

## Rationale

slic3r is one of the best open source slicers out there, and has a big, high-quality codebase. It deserves to be the point of reference for open source slicing, to test new ideas and add new features.

But it has a quite big issue: perl.

Please don't get me wrong: looking at slic3r's codebase, I can see how perl has great tools to seamlessly integrate a C++ codebase into a scripting language. In that regard, it was probably a good choice of language.

But perl is dying. Casual developers migrated to other languages long ago, and it is becoming increasingly rare to find perl monks to maintain existing codebases.

You only have to look at slic3r: it is one of the flagship open source slicers out there, with a massive user base. But slic3r's GitHub commit history shows us that almost all of the development effort is done by alexrj. With such a popular project, I would expect a large developer community. Again, don't get me wrong: alexrj is clearly up to the task.

But I am not. I can read perl, but I am not at ease writing it - at all. As most other people, I really prefer python.

As I intend to build on slic3r to test some new ideas in slicing, this is a problem for me. So I am starting this as a testbed for a python library exposing the core C++ slic3r code. I only need a few bits of slic3r for my work, but hopefully it can be used as a framework to develop a full-fledged python library.

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

