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

import sys
import os
import shutil

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


import os
from distutils.sysconfig import get_config_vars

#disable these horribly annoying warnings
(opt,) = get_config_vars('OPT')
opt += ' -Wno-unused-local-typedefs'
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

pyname = "_pyslic3r"

# clean previous build
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if (name.startswith(pyname) and not(name.endswith(".pyx") or name.endswith(".pxd"))):
            os.remove(os.path.join(root, name))
    for name in dirs:
        if (name == "build"):
            shutil.rmtree(name)

incp = "-I../deps/Slic3r/Slic3r/xs/src/"
incs = [incp+x for x in ("", "admesh/", "boost/", "poly2tri/", "libscli3r/")]

cflags = []

# build "pyslic3r.so" python extension to be added to "PYTHONPATH" afterwards...
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension(pyname, 
                  sources=[pyname+".pyx"],
                  #libraries=["slic3rlib"],          # to link dynamically: refers to "libslic3rlib.so"
                  language="c++",                   # remove this if C and not C++
                  extra_compile_args=incs+cflags,#+["-fopenmp", "-O3"],
                  extra_objects=["../deps/Slic3r/Slic3r-build/libslic3rlib.a"], #to link statically
                  extra_link_args=[#"-L./", #to link dynamically: path to "libslic3rlib.so"
                                   #"-DSOME_DEFINE_OPT", 
                                   #"-L./some/extra/dependency/dir/"
                                   ]
             )
        ]
)           