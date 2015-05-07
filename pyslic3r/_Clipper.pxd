cimport Clipper as c
from libcpp cimport bool
cimport numpy as cnp

cdef class ClipperPaths:
  cdef c.Paths *thisptr

  cpdef              reverse        (self)
  cdef  c.Paths   * _simplify       (self, c.Paths *out, c.PolyFillType fillType=*) nogil
  cpdef ClipperPaths simplify       (self, int fillType=*)
  cpdef              simplifyInPlace(self, int fillType=*)
  cdef  c.Paths   * _clean          (self, c.Paths *out, double distance=*) nogil
  cpdef ClipperPaths clean          (self, double distance=*)
  cpdef              cleanInPlace   (self, double distance=*)
  cpdef unsigned int numPaths       (self)
  cpdef bool         orientation    (self, unsigned int npath)
  cpdef double       area           (self, unsigned int npath)
  cpdef int          pointInPolygon (self, unsigned int npath, int x, int y)
  cpdef cnp.ndarray                        orientations(self)
  cpdef cnp.ndarray[cnp.float64_t, ndim=1] areas       (self)

cdef class ClipperPolyTree:
  cdef c.PolyTree *thisptr
  cdef c.Paths *toPaths(self, c.Paths *output=*)
  cpdef ClipperPaths toClipperPaths      (self)
  cpdef ClipperPaths toClipperPathsByType(self, bool closed)

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
