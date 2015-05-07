#cython: embedsignature=True

cimport cython
from libcpp cimport bool

cimport Clipper as c

cimport numpy as cnp
import numpy as np

from libc.stdio cimport *


#np.import_array()
cnp.import_array()

cdef extern from "numpy/ndarraytypes.h" nogil:
  int NPY_ARRAY_CARRAY

cdef extern from "math.h" nogil:
  double fabs(double)

#enum ClipType
ctIntersection      = c.ctIntersection
ctUnion             = c.ctUnion
ctDifference        = c.ctDifference
ctXor               = c.ctXor
#enum PolyType
ptSubject           = c.ptSubject
ptClip              = c.ptClip
#enum PolyFillType
pftEvenOdd          = c.pftEvenOdd
pftNonZero          = c.pftNonZero
pftPositive         = c.pftPositive
pftNegative         = c.pftNegative
#enum InitOptions
ioReverseSolution   = c.ioReverseSolution
ioStrictlySimple    = c.ioStrictlySimple
ioPreserveCollinear = c.ioPreserveCollinear
#enum JoinType
jtSquare            = c.jtSquare
jtRound             = c.jtRound
jtMiter             = c.jtMiter
#enum EndType
etClosedPolygon     = c.etClosedPolygon
etClosedLine        = c.etClosedLine
etOpenButt          = c.etOpenButt
etOpenSquare        = c.etOpenSquare
etOpenRound         = c.etOpenRound



cdef class ClipperPaths:
  """Thin wrapper around Clipper::Paths. It is intended just as a temporary object
  to do Clipper operations on the data, the end results to be incorporated back
  into a SlicedModel"""

  def __cinit__(self):       self.thisptr = new c.Paths()
  def __dealloc__(self): del self.thisptr
  
  cpdef reverse(self):
    c.ReversePaths(self.thisptr[0])
  
  cdef c.Paths * _simplify(self, c.Paths *out, c.PolyFillType fillType=c.pftEvenOdd) nogil:
    if out==NULL:
      out = new c.Paths()
    c.SimplifyPolygons(self.thisptr[0], out[0], fillType)
    return out
  
  cpdef ClipperPaths simplify(self, int fillType=c.pftEvenOdd):
    cdef ClipperPaths out = ClipperPaths()
    out.thisptr = self._simplify(out.thisptr, <c.PolyFillType>fillType)
    return out

  cpdef simplifyInPlace(self, int fillType=c.pftEvenOdd):
    c.SimplifyPolygons(self.thisptr[0], <c.PolyFillType>fillType)
  
  cdef c.Paths * _clean(self, c.Paths *out, double distance=1.415) nogil:
    if out==NULL:
      out = new c.Paths()
    c.CleanPolygons(self.thisptr[0], out[0], distance)
    return out
  
  cpdef ClipperPaths clean(self, double distance=1.415):
    cdef ClipperPaths out = ClipperPaths()
    out.thisptr = self._clean(out.thisptr, distance)
    return out

  cpdef cleanInPlace(self, double distance=1.415):
    c.CleanPolygons(self.thisptr[0], distance)
  
  cpdef unsigned int numPaths(self):
    return self.thisptr[0].size()
  
  cpdef bool orientation(self, unsigned int npath):
    return c.Orientation(self.thisptr[0][npath])
    
  @cython.boundscheck(False)
  cpdef cnp.ndarray orientations(self):
    cdef cnp.npy_intp length = self.thisptr[0].size()
    #using cnp.uint8_t is an ugly hack, but there is no cnp.bool_t
    cdef cnp.ndarray out = cnp.PyArray_EMPTY(1, &length, cnp.NPY_BOOL, 0)
    cdef unsigned int k
    for k in range(length):
      out[k] = c.Orientation(self.thisptr[0][k])
    return out
  
  cpdef double area(self, unsigned int npath):
    return c.Area(self.thisptr[0][npath])
    
  @cython.boundscheck(False)
  cpdef cnp.ndarray[cnp.float64_t, ndim=1] areas(self):
    cdef cnp.npy_intp length = self.thisptr[0].size()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = cnp.PyArray_EMPTY(1, &length, cnp.NPY_FLOAT64, 0)
    cdef unsigned int k
    for k in range(length):
      out[k] = c.Area(self.thisptr[0][k])
    return out

  cpdef int pointInPolygon(self, unsigned int npath, int x, int y):
    cdef c.IntPoint p
    p.X = x
    p.Y = y
    return c.PointInPolygon(p, self.thisptr[0][npath])
  
cdef class ClipperPolyTree:
  """Thin wrapper around Clipper::PolyTree. It is intended just as a temporary object
  to do Clipper operations on the data, the end results to be incorporated back
  into a SlicedModel"""

  def __cinit__(self):       self.thisptr = new c.PolyTree()
  def __dealloc__(self): del self.thisptr

  cdef c.Paths *toPaths(self, c.Paths *output=NULL):
    if output==NULL:
      output = new c.Paths()
    else:
      output[0].clear()
    c.PolyTreeToPaths(self.thisptr[0], output[0])
    return output

  cpdef ClipperPaths toClipperPaths(self):
    cdef ClipperPaths paths = ClipperPaths()
    paths.thisptr           = self.toPaths(paths.thisptr)
    return paths
  
  cpdef ClipperPaths toClipperPathsByType(self, bool closed):
    cdef ClipperPaths paths = ClipperPaths()
    if closed:
      c.ClosedPathsFromPolyTree(self.thisptr[0], paths.thisptr[0])
    else:
      c.OpenPathsFromPolyTree(self.thisptr[0], paths.thisptr[0])
    return paths
    


cdef class ClipperClip:
  """Thin wrapper around Clipper::Clipper, to do operations with ClipperPaths"""

  property reverseSolution:
    def __get__(self):    return self.thisptr[0].ReverseSolution()
    def __set__(self, bool val): self.thisptr[0].ReverseSolution(val)

  property strictlySimple:
    def __get__(self):    return self.thisptr[0].StrictlySimple()
    def __set__(self, bool val): self.thisptr[0].StrictlySimple(val)

  property preserveCollinear:
    def __get__(self):    return self.thisptr[0].PreserveCollinear()
    def __set__(self, bool val): self.thisptr[0].PreserveCollinear(val)

  property subjectFillType:
    def __get__(self):   return self.subjectfill
    def __set__(self, int val): self.subjectfill = <c.PolyFillType>val

  property clipFillType:
    def __get__(self):   return self.clipfill
    def __set__(self, int val): self.clipfill    = <c.PolyFillType>val

  property clipType:
    def __get__(self):   return self.cliptype
    def __set__(self, int val): self.cliptype    = <c.ClipType>val

  def __cinit__  (self):
    self.thisptr = new c.Clipper()
    self.subjectfill = c.pftEvenOdd
    self.clipfill    = c.pftEvenOdd
    self.cliptype    = c.ctIntersection
  def __dealloc__(self): del self.thisptr

  cpdef bool AddSubjects(self, ClipperPaths paths, bool pathsAreClosed=True):
    return self.thisptr[0].AddPaths(paths.thisptr[0], c.ptSubject, pathsAreClosed)
  cpdef bool AddClips   (self, ClipperPaths paths):
    return self.thisptr[0].AddPaths(paths.thisptr[0], c.ptClip,    True)
  
  cpdef Clear(self):
    self.thisptr[0].Clear()

  cpdef bool ExecuteWithPaths   (self, ClipperPaths    solution=None):
    if solution is None:
      solution = ClipperPaths()
    return self.thisptr[0].Execute(<c.ClipType>self.clipType, solution.thisptr[0], self.subjectfill, self.clipfill)

  cpdef bool ExecuteWithPolyTree(self, ClipperPolyTree solution=None):
    if solution is None:
      solution = ClipperPaths()
    return self.thisptr[0].Execute(<c.ClipType>self.clipType, solution.thisptr[0], self.subjectfill, self.clipfill)
    
  cpdef bool Execute            (self, object          solution=None):
    if   solution is None:                      return self.ExecuteWithPolyTree()
    elif isinstance(solution, ClipperPaths):    return self.ExecuteWithPaths   (<ClipperPaths>   solution)
    elif isinstance(solution, ClipperPolyTree): return self.ExecuteWithPolyTree(<ClipperPolyTree>solution)
    return False



cdef class ClipperOffset:
  """Thin wrapper around Clipper::ClipperOffset, to do operations with ClipperPaths"""

  property miterLimit:
    def __get__(self):      return self.thisptr[0].MiterLimit
    def __set__(self, double val): self.thisptr[0].MiterLimit   = val

  property arcTolerance:
    def __get__(self):      return self.thisptr[0].ArcTolerance
    def __set__(self, double val): self.thisptr[0].ArcTolerance = val

  property delta:
    def __get__(self):    return self.delta
    def __set__(self, bool val): self.delta = val

  def __cinit__  (self, double delta=1.0):
    self.thisptr = new c.ClipperOffset()
    self.delta   = delta
  def __dealloc__(self): del self.thisptr

  cpdef AddPaths(self, ClipperPaths paths, int joinType=c.jtRound, int endType=c.etOpenRound):
    self.thisptr[0].AddPaths(paths.thisptr[0], <c.JoinType>joinType, <c.EndType>endType)

  cpdef Clear(self):
    self.thisptr[0].Clear()

  cpdef ExecuteWithPaths   (self, ClipperPaths    solution=None):
    if solution is None:
      solution = ClipperPaths()
    self.thisptr[0].Execute(solution.thisptr[0], self.delta)

  cpdef ExecuteWithPolyTree(self, ClipperPolyTree solution=None):
    if solution is None:
      solution = ClipperPolyTree()
    self.thisptr[0].Execute(solution.thisptr[0], self.delta)

  cpdef Execute            (self, object          solution=None):
    if   solution is None:                      return self.ExecuteWithPolyTree()
    elif isinstance(solution, ClipperPaths):    return self.ExecuteWithPaths   (<ClipperPaths>   solution)
    elif isinstance(solution, ClipperPolyTree): return self.ExecuteWithPolyTree(<ClipperPolyTree>solution)
    return False


  