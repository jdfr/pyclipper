cimport Clipper as c
from libcpp cimport bool

cdef class ClipperPaths:
  cdef c.Paths *thisptr

cdef class ClipperPolyTree:
  cdef c.PolyTree *thisptr
  cdef c.Paths *toPaths(self, c.Paths *output=*)
  cpdef ClipperPaths toClipperPaths(self)

cdef class ClipperClip:
  cdef c.Clipper     *thisptr
  cdef c.PolyFillType subjectfill
  cdef c.PolyFillType clipfill
  cdef c.ClipType     cliptype
  cpdef bool AddSubjects(self, ClipperPaths paths, bool pathsAreClosed=*)
  cpdef bool AddClips   (self, ClipperPaths paths)
  cpdef Clear(self)
  cpdef bool Execute            (self, object          solution=*)
  cpdef bool ExecuteWithPaths   (self, ClipperPaths    solution=*)
  cpdef bool ExecuteWithPolyTree(self, ClipperPolyTree solution=*)

cdef class ClipperOffset:
  cdef c.ClipperOffset *thisptr
  cdef double delta
  cpdef AddPaths(self, ClipperPaths paths, int joinType=*, int endType=*)
  cpdef Clear(self)
  cpdef Execute            (self, object          solution=*)
  cpdef ExecuteWithPaths   (self, ClipperPaths    solution=*)
  cpdef ExecuteWithPolyTree(self, ClipperPolyTree solution=*)
