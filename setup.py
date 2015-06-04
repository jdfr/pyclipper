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

#for this script to work properly in LINUX, the Slic3r c++ libraries must have been compiled in ./deps/Slic3r/cmakebuild

import os
import os.path as op
import shutil

from distutils.core      import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars

from Cython.Distutils import build_ext
import numpy as n

WINDOWS       = os.name=='nt'
standalone    = __name__ == "__main__"
dirname       = "pyslic3r"
slic3rsrcdir  = "deps/Slic3r/Slic3r/xs/src"

if WINDOWS:

  description = "Python wrapper for ClipperLib library"

  if standalone:
    setup(
      name        = dirname,
      cmdclass    = {'build_ext': build_ext},
      description = description,
      packages    = [dirname],
      ext_modules = [
        Extension(
            dirname+".Clipper", 
            sources      = [dirname+"/Clipper.pyx", op.join(slic3rsrcdir, 'clipper.cpp')],
            language     = "c++",
            include_dirs = [slic3rsrcdir, n.get_include()])])

else:

  #disable these horribly annoying warnings
  (opt,) = get_config_vars('OPT')
  opt   += ' -Wno-unused-local-typedefs -Wno-unused-function -Wno-cpp'
  opt    = " ".join(
      flag for flag in opt.split() if flag != '-Wstrict-prototypes'
  )

  description       =  "Python wrapper for Scli3r C++ library"
  pynames           = ["Clipper", "SlicedModel", "TriangleMesh"]
  libnamesLD        = ["clipper", "slic3rlib"]
  
  libnamesdir       = ["%s/%s"                       % (dirname, libnameLD) for libnameLD in libnamesLD]
  libnamesreal      = ['lib%s.so'                    %           libnameLD  for libnameLD in libnamesLD]
  libnamesOrig      = ['./deps/Slic3r/cmakebuild/%s' %           libname    for libname   in libnamesreal]
  libnamesDest      = ['./%s/%s'                     % (dirname, libname)   for libname   in libnamesreal]

  runtimelibdirs    = ["$ORIGIN"]
  
  includepaths      = ["%s/%s"%(slic3rsrcdir,x) for x in ("", "libscli3r/")]

  if standalone:
    os.environ['OPT'] = opt
    
    for origpath, destpath in zip(libnamesOrig, libnamesDest):
      shutil.copyfile(origpath, destpath)
    
    setup(
        name          = dirname,
        cmdclass      = {'build_ext': build_ext},
        description   = description,
        packages      = [dirname],
        ext_modules   = [
          Extension(
            "%s.%s" % (dirname, name), 
            sources              = ["%s/%s.pyx" % (dirname, name)],
            libraries            = libnamesdir,
            runtime_library_dirs = runtimelibdirs,
            language             = "c++",
            include_dirs         = includepaths,
            package_data         = {dirname: libnamesreal}
          )
          for name in pynames]
    )           
