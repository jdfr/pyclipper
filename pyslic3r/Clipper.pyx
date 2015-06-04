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

#cython: embedsignature=True

cimport cython
from libcpp cimport bool

cimport Clipper_defs as c

cimport numpy as cnp
import  numpy as  np

cimport libc.stdio as io

from libc.math cimport INFINITY, NAN#, isnan

cdef extern from "<cmath>" namespace "std":
  bint isnan(double x) nogil

#np.import_array()
cnp.import_array()

cdef extern from "numpy/ndarraytypes.h" nogil:
  int NPY_ARRAY_CARRAY

cimport cpython.ref as ref

import sys
import os
Windows = os.name=='nt'
if Windows:
  import msvcrt

cdef extern from "Python.h":
    ctypedef struct FILE
    FILE* PyFile_AsFile(object)

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

cdef class ClipperPathsIterator:
  cdef ClipperPaths paths
  cdef size_t current
  
  def __cinit__(self, ClipperPaths p):
    self.paths   = p
    self.current = 0
    
  @cython.boundscheck(False)
  def __next__(self):
    if self.current >= self.paths.thisptr[0].size():
      raise StopIteration
    else:
      x = Path2arrayView(self.paths, &self.paths.thisptr[0][self.current])
      self.current += 1
      return x


@cython.boundscheck(False)
def arrayListToClipperPaths(list paths, ClipperPaths output=None):
  """Convert a list of two-dimensional arrays of type int64 to a ClipperPaths 
  object (this list can be got with list(x for x in paths)"""
  cdef c.Path *path
  cdef size_t k, p, npaths, npoints
  cdef cnp.ndarray[cnp.int64_t, ndim=1] array
  if output is None:
    output = ClipperPaths()
  npaths = len(paths)
  output.thisptr[0].resize(npaths)
  for k in range(npaths):
    array   = paths[k]
    npoints = array.shape[0]
    path    = &output.thisptr[0][k]
    path[0].resize(npoints)
    for p in range(npoints):
      path[0][p].X = array[p,0]
      path[0][p].Y = array[p,1]
    

cdef class ClipperPaths:
  """Thin wrapper around Clipper::Paths. It is intended just as a temporary object
  to do Clipper operations on the data, the end results to be incorporated back
  into another kind of object"""

  def __cinit__(self):       self.thisptr = new c.Paths()
  def __dealloc__(self): del self.thisptr

  def __reduce__(self):
    d = {'state': list(self.__iter__())}
    return (ClipperPaths, (), d)
  def __setstate__(self, d):
    arrayListToClipperPaths(d['state'], self)
    
  def __len__(self):
    return self.thisptr[0].size()
  def __iter__(self):
    return ClipperPathsIterator(self)
  
  def __getitem__(self, val):
    """Basic indexing support"""
    cdef int npath
    if   isinstance(val, int):
      npath = val
      if (npath<0) or (<size_t>npath>=self.thisptr[0].size()):
        raise Exception('Invalid index')
      return Path2arrayView(self, &self.thisptr[0][npath])
    elif isinstance(val, slice) or isinstance(val, tuple):
      raise Exception('This object does not support slicing, only indexing')
    else:
      raise IndexError('Invalid slice object')
  
  def clear(self):
    self.thisptr[0].clear()
  
  def copyFrom(self, ClipperPaths other):
    self.thisptr[0] = other.thisptr[0]
    
  def copy(self):
    cdef ClipperPaths out = ClipperPaths()
    out.thisptr[0] = self.thisptr[0]
    return out

  def __copy__(self): return self.copy()

  @cython.boundscheck(False)
  def getBoundingBox(self):  
    """Compute the bounding box."""
    cdef size_t k1, k2
    cdef double minx, maxx, miny, maxy, x, y
    minx = miny =  INFINITY
    maxx = maxy = -INFINITY
    for k1 in range(self.thisptr[0].size()):
      for k2 in range(self.thisptr[0][k1].size()):
        x = self.thisptr[0][k1][k2].X
        y = self.thisptr[0][k1][k2].Y
        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)
    return (minx, maxx, miny, maxy)

  @cython.boundscheck(False)
  def reverse(self, int index=-1):
    if index<0:
      c.ReversePaths(self.thisptr[0])
    else:
      if (<unsigned int>index)>=self.thisptr[0].size():
        raise Exception('Invalid index')
      c.ReversePath(self.thisptr[0][index])
  
  def simplify(self, int fillType=c.pftEvenOdd, ClipperPaths out=None):
    if out is None:
      out = ClipperPaths()
    else:
      out.thisptr[0].clear()
    c.SimplifyPolygons(self.thisptr[0], out.thisptr[0], <c.PolyFillType>fillType)
    return out

  def simplifyInPlace(self, int fillType=c.pftEvenOdd):
    c.SimplifyPolygons(self.thisptr[0], <c.PolyFillType>fillType)
  
  def clean(self, double distance=1.415, ClipperPaths out=None):
    if out is None:
      out = ClipperPaths()
    else:
      out.thisptr[0].clear()
    c.CleanPolygons(self.thisptr[0], out.thisptr[0], distance)
    return out

  cdef toFileObject(self, io.FILE *f):
    """low level write function"""
    cdef size_t numpaths = self.thisptr[0].size()
    cdef size_t k, i, np, bytesize, count
    cdef c.IntPoint * p
    count = io.fwrite(&numpaths, sizeof(size_t), 1, f)
    if count!=1: raise IOError
    for k in range(numpaths):
      np = self.thisptr[0][k].size()
      count = io.fwrite(&np, sizeof(size_t), 1, f)
      for i in range(np):
        p = &self.thisptr[0][k][i]
        count = io.fwrite(&p[0].X, sizeof(c.cInt), 1, f)
        if count!=1: raise IOError
        count = io.fwrite(&p[0].Y, sizeof(c.cInt), 1, f)
        if count!=1: raise IOError
      
  cdef fromFileObject(self, io.FILE *f):
    """low level read function"""
    cdef size_t count, numpaths, k, i, np, bytesize
    cdef c.IntPoint * p
    count = io.fread(&numpaths, sizeof(size_t), 1, f)
    if count!=1: raise IOError
    self.thisptr[0].clear()
    self.thisptr[0].resize(numpaths)
    for k in range(numpaths):
      count = io.fread(&np, sizeof(size_t), 1, f)
      if count!=1: raise IOError
      self.thisptr[0][k].resize(np)
      for i in range(np):
        p = &self.thisptr[0][k][i]
        count = io.fread(&p[0].X, sizeof(c.cInt), 1, f)
        if count!=1: raise IOError
        count = io.fread(&p[0].Y, sizeof(c.cInt), 1, f)
        if count!=1: raise IOError

  def toStream(self, stream):
    """write in binary mode. If stream is a string, it is the name of the file to
    write to. If it is None, data will be written to standard output. Otherwise,
    it must be a file object. The stream is changed to binary mode, if precise"""
    cdef File f = File(stream, 'wb', True)
    self.toFileObject(f.f)
    f.close()
    
  def fromStream(self, stream):
    """read in binary mode. If stream is a string, it is the name of the file to
    read from. If it is None, data will be read from standard output. Otherwise,
    it must be a file object. The stream is changed to binary mode, if precise"""
    cdef File f = File(stream, 'rb', False)
    self.fromFileObject(f.f)
    f.close()
    
  def   cleanInPlace(self, double distance=1.415):
    c.CleanPolygons(self.thisptr[0], distance)
  
  @cython.boundscheck(False)
  def orientation(self, size_t npath):
    if npath>=self.thisptr[0].size():
      raise Exception('Invalid index')
    return c.Orientation(self.thisptr[0][npath])
    
  @cython.boundscheck(False)
  def orientations(self):
    cdef cnp.npy_intp length = self.thisptr[0].size()
    #using cnp.uint8_t is an ugly hack, but there is no cnp.bool_t
    cdef cnp.ndarray out = cnp.PyArray_EMPTY(1, &length, cnp.NPY_BOOL, 0)
    cdef size_t k
    for k in range(<size_t>length):
      out[k] = c.Orientation(self.thisptr[0][k])
    return out
  
  @cython.boundscheck(False)
  def area(self, size_t npath):
    if npath>=self.thisptr[0].size():
      raise Exception('Invalid index')
    return c.Area(self.thisptr[0][npath])
    
  @cython.boundscheck(False)
  def areas(self):
    cdef cnp.npy_intp length = self.thisptr[0].size()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = cnp.PyArray_EMPTY(1, &length, cnp.NPY_FLOAT64, 0)
    cdef size_t k
    for k in range(<size_t>length):
      out[k] = c.Area(self.thisptr[0][k])
    return out

  @cython.boundscheck(False)
  def pointInPolygon(self, size_t npath, c.cInt x, c.cInt y):
    if npath>=self.thisptr[0].size():
      raise Exception('Invalid index')
    cdef c.IntPoint p
    p.X = x
    p.Y = y
    return c.PointInPolygon(p, self.thisptr[0][npath])

    
cdef class File:
  """Custom and (hopefully) fast lightweight wrapper for files.
  It works with strings representing file names, or stdin/stdout
  if None is provided (forcing them to binary mode)"""
  
  def __cinit__(self, object stream, str mode="wb", bool write=True):
    """open a FILE* object"""
    cdef fileno
    self.write     = write
    if   isinstance(stream, basestring):
      self.doclose = True
      self.f       = io.fopen(stream, mode)
    elif stream is None:
      self.doclose = False
      if write:
        stream     = sys.stdout #f = io.stdout
      else:
        stream     = sys.stdin  #f = io.stdin
      fileno       = stream.fileno()
      if Windows:
        msvcrt.setmode(fileno, os.O_BINARY)
      self.f       = io.fdopen(fileno, mode)
    else:
      raise IOError("This class can only open new files or reopen stdin/stdout")
  
  cdef void close(self):
    """close the file just once, if it has to be done"""
    if self.doclose:
      io.fclose(self.f)
      self.doclose = False
  
  def __dealloc__(self):
    """make sure it is closed"""
    self.close()
    
    
def ClipperPathsAndZsToStream(c.cInt npths, object pathsAndZss, object stream):
  """from a sequence of pairs (ClipperPaths,zs), where zs is itself a sequence of double values,
  write to a filename, a file object or stdout (if stream is None)"""
  cdef File         f = File(stream, 'wb', True)
  cdef double       z
  cdef ClipperPaths paths
  cdef c.cInt       numz
  if     io.fwrite(&npths, sizeof(npths),  1, f.f)!=1: raise IOError
  for paths,zs in pathsAndZss:
    if not hasattr(zs, '__len__'):
      zs    = (zs,)
    numz    = len(zs)
    if   io.fwrite(&numz,  sizeof(numz),   1, f.f)!=1: raise IOError
    for z in zs:
      if io.fwrite(&z,     sizeof(double), 1, f.f)!=1: raise IOError
    paths.toFileObject(f.f)
  f.close()
  
@cython.boundscheck(False)
def ClipperPathsAndZsFromStream(object stream, bool alsoYieldNumPaths=False):
  """for a file or a stream (wrapping a filename, a file-like object or stdin if 
  stream is None), make a generator yielding pairs (ClipperPaths,zs), where zs is a 
  sequence of doubles. Optionally, the first yielded object is the number of pairs"""
  cdef File         f = File(stream, 'rb', False)
  cdef double       z
  cdef cnp.ndarray  zs
  cdef c.cInt       npths, numz
  cdef int          k, m
  cdef ClipperPaths paths
  if     io.fread(&npths, sizeof(npths),  1, f.f)!=1: raise IOError
  if alsoYieldNumPaths: yield npths
  for k in range(npths):
    if   io.fread(&numz,  sizeof(numz),   1, f.f)!=1: raise IOError
    zs      = np.empty((numz,))
    for m in range(numz):
      if io.fread(&z,     sizeof(double), 1, f.f)!=1: raise IOError
      zs[m] = z
    paths = ClipperPaths()
    paths.fromFileObject(f.f)
    yield (paths,zs)
  f.close()
  
    
cdef class ClipperPolyTree:
  """Thin wrapper around Clipper::PolyTree. It is intended just as a temporary object
  to do Clipper operations on the data, the end results to be incorporated back
  into another kind of object"""

  def __cinit__(self):       self.thisptr = new c.PolyTree()
  def __dealloc__(self): del self.thisptr

  def clear(self):
    self.thisptr[0].Clear()
    
  cdef c.Paths *toPaths(self, c.Paths *output=NULL):
    if output==NULL:
      output = new c.Paths()
    else:
      output[0].clear()
    c.PolyTreeToPaths(self.thisptr[0], output[0])
    return output

  def toClipperPaths(self, ClipperPaths paths=None):
    if paths is None:
      paths                 = ClipperPaths()
    else:
      paths.thisptr[0].clear()
    paths.thisptr           = self.toPaths(paths.thisptr)
    return paths
  
  def toClipperPathsByType(self, bool closed):
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

  def __cinit__  (self, int clipType=c.ctIntersection, int subjectFillType=c.pftEvenOdd, int clipFillType=c.pftEvenOdd, bool reverseSolution=False, bool strictlySimple=False, bool preserveCollinear=False, *args, **kwargs):
    self.thisptr = new c.Clipper()
    self.subjectfill = <c.PolyFillType>subjectFillType
    self.clipfill    = <c.PolyFillType>clipFillType
    self.cliptype    = <c.ClipType>    clipType
    if strictlySimple:
      self.thisptr[0].StrictlySimple   (True)
    if reverseSolution:
      self.thisptr[0].ReverseSolution  (True)
    if preserveCollinear:
      self.thisptr[0].PreserveCollinear(True)
    
  def __dealloc__(self): del self.thisptr

  cpdef bool AddPaths   (self, ClipperPaths paths, c.PolyType typ, bool pathsAreClosed=True): return self.thisptr[0].AddPaths(paths.thisptr[0], typ,         pathsAreClosed)
  def        AddSubjects(self, ClipperPaths paths,                 bool pathsAreClosed=True): return self.           AddPaths(paths,            c.ptSubject, pathsAreClosed)
  def        AddClips   (self, ClipperPaths paths                                          ): return self.           AddPaths(paths,            c.ptClip,    True)
  
  def clear(self):
    self.thisptr[0].Clear()
  
  #these two methods are defined just in case we want to profile
  cdef bool ExecuteP (self, c.Paths    *solution, c.ClipType clipType, c.PolyFillType subjectFillType, c.PolyFillType clipFillType): return self.thisptr[0].Execute(clipType, solution[0], subjectFillType, clipFillType)
  cdef bool ExecutePT(self, c.PolyTree *solution, c.ClipType clipType, c.PolyFillType subjectFillType, c.PolyFillType clipFillType): return self.thisptr[0].Execute(clipType, solution[0], subjectFillType, clipFillType)

  cpdef      Execute(self, object solution, int clipType=-1, int subjectFillType=-1, int clipFillType=-1):
    if        clipType<0:        clipType = self.cliptype
    if subjectFillType<0: subjectFillType = self.subjectfill
    if    clipFillType<0:    clipFillType = self.clipfill
    if   isinstance(solution, ClipperPaths   ): return self.ExecuteP ((<ClipperPaths>   solution).thisptr, <c.ClipType>clipType, <c.PolyFillType>subjectFillType, <c.PolyFillType>clipFillType)
    elif isinstance(solution, ClipperPolyTree): return self.ExecutePT((<ClipperPolyTree>solution).thisptr, <c.ClipType>clipType, <c.PolyFillType>subjectFillType, <c.PolyFillType>clipFillType)
    else                                      : raise ValueError('Object of incorrect type: '+type(solution))

  def do(self, object result, int operation, ClipperPaths subject, ClipperPaths clip):
    """Simple, straightforward use of the engine. Assumes that subject and
    clip are closed paths, and that the engine is clear. This method clears the
    before returning"""
    cdef bool out #avoid stupid C++ warning arising from cython's code generation
    out = self.thisptr[0].AddPaths(subject.thisptr[0], c.ptSubject, True)
    out = self.thisptr[0].AddPaths(clip   .thisptr[0], c.ptClip,    True)
    self.Execute (result, clipType=operation)
    self.thisptr[0].Clear()
    return out

cdef class ClipperOffset:
  """Thin wrapper around Clipper::ClipperOffset, to do operations with ClipperPaths"""

  property miterLimit:
    def __get__(self):      return self.thisptr[0].MiterLimit
    def __set__(self, double val): self.thisptr[0].MiterLimit   = val

  property arcTolerance:
    def __get__(self):      return self.thisptr[0].ArcTolerance
    def __set__(self, double val): self.thisptr[0].ArcTolerance = val

  property delta:
    def __get__(self):      return self._delta
    def __set__(self, double val): self._delta = val

  property joinType:
    def __get__(self):      return self.jointype
    def __set__(self, int    val): self.jointype = <c.JoinType>val

  property endType:
    def __get__(self):      return self.endtype
    def __set__(self, int    val): self.endtype  = <c.EndType> val

  def __cinit__  (self, double miterLimit=3.0, double arcTolerance=3.0, double delta=1.0, int joinType=c.jtRound, int endType=c.etClosedPolygon, *args, **kwargs):
    self.thisptr                 = new c.ClipperOffset()
    self._delta                  = delta
    self.jointype                = <c.JoinType>joinType
    self.endtype                 = <c.EndType>endType
    self.thisptr[0].MiterLimit   = miterLimit
    self.thisptr[0].ArcTolerance = arcTolerance
  def __dealloc__(self): del self.thisptr

  def AddPaths(self, ClipperPaths paths, int joinType=-1, int endType=-1):
    if joinType<0: joinType = self.jointype
    if  endType<0:  endType = self. endtype
    self.thisptr[0].AddPaths(paths.thisptr[0], <c.JoinType>joinType, <c.EndType>endType)

  def clear(self):
    self.thisptr[0].Clear()

  #these two methods are defined just in case we want to profile
  cdef void ExecuteP (self, c.Paths    *solution, double delta): self.thisptr[0].Execute(solution[0], delta)
  cdef void ExecutePT(self, c.PolyTree *solution, double delta): self.thisptr[0].Execute(solution[0], delta)
  
  cpdef Execute(self, object solution, double delta=NAN):
    if isnan(delta): delta = self._delta
    if   isinstance(solution, ClipperPaths   ): self.ExecuteP ((<ClipperPaths>   solution).thisptr, delta)
    elif isinstance(solution, ClipperPolyTree): self.ExecutePT((<ClipperPolyTree>solution).thisptr, delta)
    else                                      : raise ValueError('Object of incorrect type: '+type(solution))

  cpdef do(self, object output, double delta, ClipperPaths inputs):
    """Simple, straightforward use of the engine. Assumes that the engine is
    clear, and clears it before returning"""
    self.thisptr[0].AddPaths(inputs.thisptr[0], self.jointype, self.endtype)
    self.Execute(output, delta)
    self.thisptr[0].Clear()

  def do2(self, object output, double delta1, double delta2, ClipperPaths inputs, ClipperPaths intermediate):
    """Simple, straightforward use of the engine. Assumes that the engine is
    clear, and clears it before returning"""
    self.do(intermediate, delta1, inputs)
    self.do(output,       delta2, intermediate)


#strides have to be computed just once
cdef cnp.npy_intp *pointstrides = [sizeof(c.IntPoint),
                                   <cnp.uint8_t*>&(<c.IntPoint*>NULL).Y - 
                                   <cnp.uint8_t*>&(<c.IntPoint*>NULL).X]

cdef cnp.ndarray Path2arrayView(ClipperPaths parent, c.Path *path):
  """Similar to Polygon2arrayI, but instead of allocating a full-blown array,
  the returned array is a view into the underlying data"""
  cdef void         *data  = &(path[0][0].X)
  cdef cnp.npy_intp *dims  = [path[0].size(),2]
  cdef cnp.ndarray  result = cnp.PyArray_New(np.ndarray, 2, dims, cnp.NPY_INT64, pointstrides,
                                             data, -1, NPY_ARRAY_CARRAY, <object>NULL)
  ##result.base is of type PyObject*, so no reference counting with this assignment
  result.base              = <ref.PyObject*>parent
  ref.Py_INCREF(parent) #so, make sure that "result" owns a reference to "parent"
  #ref.Py_INCREF(result)
  return result
  