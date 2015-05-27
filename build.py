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

ALSOSLIC3R = True  #link all C++ libraries
#ALSOSLIC3R = False #link just the clipper c++ library

#LD name of the C++ libraries
if ALSOSLIC3R:
  libnamesLD = ["clipper", "slic3rlib"] # to link dynamically: refers to "libslic3rlib.so"
else:
  libnamesLD = ["clipper"]

#file name of the slic3r C++ library
def instantiate_libnames(libnamesLD):
  return ['lib%s.so' % x for x in libnamesLD]


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

def extension_template(name, dirname, runtimelibdirs, libraries, includes, additional, additionalFiles):
  print "mira includes: "+str(includes)
  return Extension(
    "%s.%s" % (dirname, name), 
    sources=["%s/%s.pyx" % (dirname, name)]+additionalFiles,
    libraries=libraries,
    runtime_library_dirs=runtimelibdirs,
    language="c++",
    include_dirs=includes,
    extra_compile_args=cflags,#+["-fopenmp", "-O3"],
    extra_objects=[],#"../deps/Slic3r/Slic3r-build/libslic3rlib.a"], #to link statically
    extra_link_args=[],#"-DSOME_DEFINE_OPT", "-L./some/extra/dependency/dir/"]
    **additional
  )

def libraries_addpath(basepath, path, libnamesLD):
  for libname in instantiate_libnames(libnamesLD):
    yield '%s/%s/%s' % (basepath, path, libname)

def copy_external_libraries(basepath, libpath, dirname, libnamesLD):
  #copy Scli3r c++ library (it should have been build with cmake)
  for origpath, destpath in zip(libraries_addpath(basepath, libpath, libnamesLD),
                                libraries_addpath(basepath, dirname, libnamesLD)):
    shutil.copyfile(origpath, destpath)

def remove_external_libraries(basepath, dirname, libnamesLD):
  for removepath in libraries_addpath(basepath, dirname, libnamesLD):
    os.remove(removepath)

def dobuild(opt, basepath, includepaths, dirname, description, runtimelibdirs, libraries, pynames, additionalSetup, additionalExtensions, additionalFiles=[]):
  os.environ['OPT'] = opt
  
  # build "pyslic3r.so" python extension to be added to "PYTHONPATH" afterwards...
  setup(
      name        = dirname,
      cmdclass    = {'build_ext': build_ext},
      description = description,
      packages    = [dirname],
      ext_modules = [extension_template(name, dirname, runtimelibdirs, libraries, instantiate_includes(basepath, includepaths), additionalExtensions, additionalFiles) for name in pynames],
      **additionalSetup
  )           


def erasedircontents(basedir):
  for name in os.listdir(basedir):
    absname = op.join(basedir, name)
    if op.isfile(absname):
      os.remove(absname)
    else:
      shutil.rmtree(absname)

# clean previous build
def doClean(dirname, pynames):
  
  #remove ../build
  superbuild = op.join(dirname, '../build')
  if op.isdir(superbuild):
    shutil.rmtree(superbuild)
    
  basedir = op.abspath(op.join(dirname, '..'))
  #remove ../*.pyc
  for name in os.listdir(basedir):
    if name.endswith(".pyc"):
      os.remove(op.join(basedir, name))
  
  for name in os.listdir(dirname):
    absname  = op.join(dirname, name)
    if op.isfile(absname):
      toRemove = ( name.endswith(".pyc") or  #remove *.pyc
                   any(   name.startswith(pyname) and #remove ./*.pyc and ./{CYTHONMODULE}.{so, cpp}
                          not any(name.endswith(x) for x in [".pyx", ".pxd", ".pxi"])
                        for pyname in pynames) )
      if toRemove:
        print "Removing file "+absname
        os.remove(absname)
    #remove ./build
    elif op.isdir(absname) and (name == "build"):
      shutil.rmtree(absname)
