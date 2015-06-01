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

cimport Clipper_defs as c
from    libcpp cimport bool
cimport numpy        as cnp
cimport libc.stdio   as io

cdef class ClipperPaths:
  cdef  c.Paths   * thisptr
  cdef   toFileObject(self, io.FILE *f)
  cdef fromFileObject(self, io.FILE *f)

cdef class ClipperPolyTree:
  cdef c.PolyTree *  thisptr
  cdef c.Paths    *  toPaths(self, c.Paths *output=*)

cdef class ClipperClip:
  cdef  c.Clipper     *thisptr
  cdef  c.PolyFillType subjectfill
  cdef  c.PolyFillType clipfill
  cdef  c.ClipType     cliptype
  cpdef bool AddPaths           (self, ClipperPaths    paths, c.PolyType typ, bool pathsAreClosed=*)
  cdef  bool ExecuteP           (self, c.Paths    *solution, c.ClipType clipType, c.PolyFillType subjectFillType, c.PolyFillType clipFillType)
  cdef  bool ExecutePT          (self, c.PolyTree *solution, c.ClipType clipType, c.PolyFillType subjectFillType, c.PolyFillType clipFillType)
  cpdef      Execute            (self, object solution, int clipType=*, int subjectFillType=*, int clipFillType=*)

cdef class ClipperOffset:
  cdef  c.ClipperOffset *thisptr
  cdef  double _delta
  cdef  c.JoinType jointype
  cdef  c.EndType  endtype
  cdef void ExecuteP (self, c.Paths    *solution, double delta)
  cdef void ExecutePT(self, c.PolyTree *solution, double delta)
  cpdef Execute      (self, object solution, double delta=*)
  cpdef do           (self, object output,   double delta, ClipperPaths inputs)

  
cdef class File:
  cdef io.FILE* f
  cdef bool     doclose
  cdef bool     write
  
  cdef void close(self)
