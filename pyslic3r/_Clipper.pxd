cimport Clipper     as c
from    libcpp cimport bool
cimport numpy       as cnp
cimport libc.stdio  as io

cdef class ClipperPaths:
  cdef  c.Paths   * thisptr
  cdef   toFileObject(self, io.FILE *f)
  cdef fromFileObject(self, io.FILE *f)
  cdef   tofromStream(self, bool write, str mode, object stream)

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
