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

from libcpp.vector cimport vector
cimport Clipper_defs as c
from    libcpp cimport bool
cimport numpy        as cnp
cimport libc.stdio   as io

cdef class ClipperPaths:
  cdef  c.Paths   * thisptr
  cdef   toFileObject(self, io.FILE *f)
  cdef fromFileObject(self, io.FILE *f)

ctypedef vector[c.DoublePoint] DPath
ctypedef vector[DPath] DPaths

cdef class ClipperDPaths:
  cdef  DPaths   * thisptr
  cdef   toFileObject(self, io.FILE *f)
  cdef fromFileObject(self, io.FILE *f)

cdef class File:
  cdef io.FILE* f
  cdef bool     doclose
  cdef bool     iswrite
  
  cpdef close(self)
  cpdef flush(self)
