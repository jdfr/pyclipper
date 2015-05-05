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

import os
import shutil

from distutils.core      import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars

from Cython.Distutils import build_ext

#infrastructure to build the system. It is kept apart from setup.py in order to easily reutilize it in other setup.py files

#LD name of the slic3r C++ library
libnamesLD = ["clipper", "slic3rlib"] # to link dynamically: refers to "libslic3rlib.so"

#file name of the slic3r C++ library
libnames = lambda libnamesLD: ['lib%s.so' % x for x in libnamesLD]


#disable these horribly annoying warnings
(opt,) = get_config_vars('OPT')
opt += ' -Wno-unused-local-typedefs -Wno-unused-function -Wno-cpp'
opt = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

basepath = '.'

#*.h* include dirs for the cython modules
includeroot  = "%s/deps/Slic3r/Slic3r/xs/src/"
includepaths = [includeroot+x for x in ("", "admesh/", "boost/", "poly2tri/", "libscli3r/")]

cflags = []


def instantiate_includes(basepath, includepaths):
  return [x % basepath for x in includepaths]

def external_libraries(dirname, libnamesLD):
  return ["%s/%s" % (dirname, libnameLD) for libnameLD in libnamesLD]

def extension_template(name, dirname, libraries, includes, additional):
  return Extension(
    "%s.%s" % (dirname, name), 
    sources=["%s/%s.pyx" % (dirname, name)],
    libraries=libraries,
    language="c++",
    include_dirs=includes,
    extra_compile_args=cflags,#+["-fopenmp", "-O3"],
    extra_objects=[],#"../deps/Slic3r/Slic3r-build/libslic3rlib.a"], #to link statically
    extra_link_args=[],#"-DSOME_DEFINE_OPT", "-L./some/extra/dependency/dir/"]
    **additional
  )


def copy_external_libraries(libpath, dirname, libnamesLD):
  #copy Scli3r c++ library (it should have been build with cmake)
  for libname in libnames(libnamesLD):
    shutil.copyfile('%s/%s/%s' % (basepath, libpath, libname), '%s/%s/%s' % (basepath, dirname, libname))

def dobuild(opt, basepath, includepaths, dirname, libraries, pynames, additional):
  os.environ['OPT'] = opt
  
  # build "pyslic3r.so" python extension to be added to "PYTHONPATH" afterwards...
  setup(
      cmdclass    = {'build_ext': build_ext},
      description = "Python wrapper for Scli3r C++ library",
      ext_modules = [extension_template(name, dirname, libraries, instantiate_includes(basepath, includepaths), additional) for name in pynames]
  )           
