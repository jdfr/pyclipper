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

#infrastructure to build the system.
#It is kept apart from setup.py in order to easily reutilize it in other setup.py files

#see setup.py for guidance on how to use this infrastructure

import os
import os.path as op
import shutil

from distutils.core      import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars

from Cython.Distutils import build_ext


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

#*.h* include dirs for the cython modules
includeroot  = "%s/deps/Slic3r/Slic3r/xs/src/"
includepaths = [includeroot+x for x in ("", "libscli3r/")]

cflags = []


def instantiate_includes(basepath, includepaths):
  return [x % basepath for x in includepaths]

def external_libraries(dirname, libnamesLD):
  return ["%s/%s" % (dirname, libnameLD) for libnameLD in libnamesLD]

def extension_template(name, dirname, runtimelibdirs, libraries, includes, additional):
  return Extension(
    "%s.%s" % (dirname, name), 
    sources=["%s/%s.pyx" % (dirname, name)],
    libraries=libraries,
    runtime_library_dirs=runtimelibdirs,
    language="c++",
    include_dirs=includes,
    extra_compile_args=cflags,#+["-fopenmp", "-O3"],
    extra_objects=[],#"../deps/Slic3r/Slic3r-build/libslic3rlib.a"], #to link statically
    extra_link_args=[],#"-DSOME_DEFINE_OPT", "-L./some/extra/dependency/dir/"]
    **additional
  )


def copy_external_libraries(basepath, libpath, dirname, libnamesLD):
  #copy Scli3r c++ library (it should have been build with cmake)
  for libname in libnames(libnamesLD):
    shutil.copyfile('%s/%s/%s' % (basepath, libpath, libname), '%s/%s/%s' % (basepath, dirname, libname))

def dobuild(opt, basepath, includepaths, dirname, description, runtimelibdirs, libraries, pynames, additionalSetup, additionalExtensions):
  os.environ['OPT'] = opt
  
  # build "pyslic3r.so" python extension to be added to "PYTHONPATH" afterwards...
  setup(
      name        = dirname,
      cmdclass    = {'build_ext': build_ext},
      description = description,
      packages    = [dirname],
      ext_modules = [extension_template(name, dirname, runtimelibdirs, libraries, instantiate_includes(basepath, includepaths), additionalExtensions) for name in pynames],
      **additionalSetup
  )           

# clean previous build
def doClean(setupmodule):
  directory = setupmodule.dirname #this assumes that we are executing from dirname's base directory 
  pynames   = setupmodule.pynames #get list of extension modules from setup.py
  for name in os.listdir(directory):
    absname = op.join(directory, name)
    if op.isfile(absname):
      for pyname in pynames: 
        if (name.startswith(pyname) and not(name.endswith(".pyx") or name.endswith(".pxd"))):
          print "Removing file "+absname
          os.remove(absname)
    elif op.isdir(absname) and (name == "build"):
      shutil.rmtree(name)
