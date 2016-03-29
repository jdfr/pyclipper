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
dirname       = "pyclipper"
clipperdir    = 'clipper/clipper'

description = "Python wrapper for ClipperLib library"

if not WINDOWS and standalone:
  #disable these horribly annoying warnings
  (opt,) = get_config_vars('OPT')
  opt   += ' -Wno-unused-local-typedefs -Wno-unused-function -Wno-cpp -Wno-unused-variable'
  opt    = " ".join(
      flag for flag in opt.split() if flag != '-Wstrict-prototypes'
  )
  os.environ['OPT'] = opt

setup(
  name        = dirname,
  cmdclass    = {'build_ext': build_ext},
  description = description,
  packages    = [dirname],
  ext_modules = [
    Extension(
        dirname+".Clipper", 
        sources      = [dirname+"/Clipper.pyx", op.join(clipperdir, 'clipper.cpp')],
        language     = "c++",
        include_dirs = [clipperdir, n.get_include()])])
