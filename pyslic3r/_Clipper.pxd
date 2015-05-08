cimport Clipper as c
from libcpp cimport bool
cimport numpy as cnp

cdef class ClipperPaths:
  cdef  c.Paths   * thisptr
  cdef  c.Paths   * _simplify       (self, c.Paths *out, c.PolyFillType fillType=*) nogil
  cdef  c.Paths   * _clean          (self, c.Paths *out, double distance=*) nogil

cdef class ClipperPolyTree:
  cdef c.PolyTree *  thisptr
  cdef c.Paths    *  toPaths(self, c.Paths *output=*)

cdef class ClipperClip:
  cdef  c.Clipper     *thisptr
  cdef  c.PolyFillType subjectfill
  cdef  c.PolyFillType clipfill
  cdef  c.ClipType     cliptype
  cpdef bool AddPaths           (self, ClipperPaths    paths, c.PolyType typ, bool pathsAreClosed=*)
  cpdef bool ExecuteWithPaths   (self, ClipperPaths    solution)
  cpdef bool ExecuteWithPolyTree(self, ClipperPolyTree solution)

cdef class ClipperOffset:
  cdef  c.ClipperOffset *thisptr
  cdef  double _delta
  cpdef ExecuteWithPaths   (self, ClipperPaths    solution)
  cpdef ExecuteWithPolyTree(self, ClipperPolyTree solution)
