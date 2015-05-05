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

#for this script to work properly, the Slic3r c++ libraries must have been compiled in ./deps/Slic3r/Slic3r-build

import build as b


basepath = '.'

#directory (and name) of the Cython library
dirname = "pyslic3r"

#path of the slic3r C++ library
libpath = 'deps/Slic3r/Slic3r-build'

#Cython modules
pynames = ["_SlicedModel", "_TriangleMesh"]

additionalIncludes = ''

libraries = b.external_libraries(dirname, b.libnamesLD)

b.copy_external_libraries(libpath, dirname, b.libnamesLD)

b.dobuild(b.opt, basepath, b.includepaths, dirname, libraries, pynames, {})

