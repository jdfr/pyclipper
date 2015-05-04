# Copyright (c) 2015 Jose David Fernandez Rodriguez
#  
# This file is distributed under the terms of the
# GNU Affero General Public License, version 3
# as published by the Free Software Foundation.
# 
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public
# License along with this file. You may obtain a copy of the License at
# http://www.gnu.org/licenses/agpl-3.0.txt

# python setup.py build_ext --inplace

#for this script to work properly, the Slic3r c++ libraries must have been compiled in ../deps/Slic3r/Slic3r-build

import os
import shutil

from distutils.core      import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars

from Cython.Distutils import build_ext

#LD name of the slic3r C++ library
libnameLD = "slic3rlib" # to link dynamically: refers to "libslic3rlib.so"

#file name of the slic3r C++ library
libname = 'lib%s.so' % libnameLD

#path of the slic3r C++ library
libpath = 'deps/Slic3r/Slic3r-build'

#directory (and name) of the Cython library
dirname = "pyslic3r"

#Cython modules
pynames = ["_SlicedModel", "_TriangleMesh"]

#copy Scli3r c++ library (it should have been build with cmake)
shutil.copyfile('./%s/%s' % (libpath, libname), './%s/%s' % (dirname, libname))

#disable these horribly annoying warnings
(opt,) = get_config_vars('OPT')
opt += ' -Wno-unused-local-typedefs -Wno-unused-function'
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

#*.h* include dirs for the cython modules
includeroot  = "-I./deps/Slic3r/Slic3r/xs/src/"
includepaths = [includeroot+x for x in ("", "admesh/", "boost/", "poly2tri/", "libscli3r/")]

cflags = []

compileargs = includepaths+cflags

extension_template = lambda name: Extension(
  "pyslic3r."+name, 
  sources=["%s/%s.pyx" % (dirname, name)],
  libraries=["%s/%s" % (dirname, libnameLD)],
  language="c++",
  extra_compile_args=compileargs,#+["-fopenmp", "-O3"],
  extra_objects=[],#"../deps/Slic3r/Slic3r-build/libslic3rlib.a"], #to link statically
  extra_link_args=[]#"-DSOME_DEFINE_OPT", "-L./some/extra/dependency/dir/"]
)


if __name__ == "__main__":
  # build "pyslic3r.so" python extension to be added to "PYTHONPATH" afterwards...
  setup(
      cmdclass    = {'build_ext': build_ext},
      description = "Python wrapper for Scli3r C++ library",
      packages    = [dirname],
      ext_modules = [extension_template(name) for name in pynames]
  )           