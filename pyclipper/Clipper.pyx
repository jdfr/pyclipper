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
  int NPY_ARRAY_FARRAY

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

DEF DEBUG = False
DEBUGMODE = DEBUG #expose the compile-time directive to python
DEBUGPATH = "debug.Clipper.pyx.log" #this can be overriden by callers

cpdef writeDebug(msg):
  """Quick and dirty internal logging facility"""
  #originally for debugging the IO workflow when executing the process without a console
  with open(DEBUGPATH, 'a') as f:
    f.write(msg)

cdef class ClipperDPathsIterator:
  cdef ClipperDPaths paths
  cdef size_t current
  
  def __cinit__(self, ClipperDPaths p):
    self.paths   = p
    self.current = 0
    
  @cython.boundscheck(False)
  def __next__(self):
    if self.current >= self.paths.thisptr[0].size():
      raise StopIteration
    else:
      x = DPath2arrayView(self.paths, &self.paths.thisptr[0][self.current])
      self.current += 1
      return x

@cython.boundscheck(False)
def arrayListToClipperDPaths(list paths, ClipperDPaths output=None):
  """Convert a list of two-dimensional arrays of type int64 to a ClipperPaths 
  object (this list can be got with list(x for x in paths)"""
  cdef DPath *path
  cdef size_t k, p, npaths, npoints
  cdef cnp.ndarray[cnp.double_t, ndim=1] array
  if output is None:
    output = ClipperDPaths()
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

cdef class ClipperDPaths:
  """Thin wrapper around Clipper::Paths. It is intended just as a temporary object
  to do Clipper operations on the data, the end results to be incorporated back
  into another kind of object"""

  def __cinit__(self):       self.thisptr = new DPaths()
  def __dealloc__(self): del self.thisptr

  def __reduce__(self):
    d = {'state': list(self.__iter__())}
    return (ClipperDPaths, (), d)
  def __setstate__(self, d):
    arrayListToClipperDPaths(d['state'], self)
    
  def __len__(self):
    return self.thisptr[0].size()
  def __iter__(self):
    return ClipperDPathsIterator(self)
  
  def __getitem__(self, val):
    """Basic indexing support"""
    cdef int npath
    if   isinstance(val, int):
      npath = val
      if (npath<0) or (<size_t>npath>=self.thisptr[0].size()):
        raise Exception('Invalid index')
      return DPath2arrayView(self, &self.thisptr[0][npath])
    elif isinstance(val, slice) or isinstance(val, tuple):
      raise Exception('This object does not support slicing, only indexing')
    else:
      raise IndexError('Invalid slice object')
  
  @cython.boundscheck(False)
  def addPath(self, cnp.ndarray[cnp.double_t, ndim=2] path):
    cdef cnp.int64_t k
    cdef size_t n = self.thisptr[0].size()
    cdef DPath *cpath
    if path.shape[1]!=2:
      raise ValueError("the path must be an array with two columns!") 
    if path.shape[0]==0:
      raise ValueError("the path cannot be empty!!!!") 
    self.thisptr[0].resize(n+1)
    cpath = &(self.thisptr[0][n])
    cpath[0].resize(path.shape[0])
    for k in range(path.shape[0]):
      cpath[0][k].X = path[k,0]
      cpath[0][k].Y = path[k,1]
  
  def clear(self):
    self.thisptr[0].clear()
  
  def copyFrom(self, ClipperDPaths other):
    self.thisptr[0] = other.thisptr[0]
    
  def copy(self):
    cdef ClipperDPaths out = ClipperDPaths()
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

  cdef toFileObject(self, io.FILE *f):
    """low level write function"""
    cdef size_t numpaths = self.thisptr[0].size()
    cdef size_t k, i, np
    cdef c.DoublePoint * p
    IF DEBUG:     writeDebug("  WRITING A CLIPPERPATH WITH %d PATHS\n" % numpaths)
    if     io.fwrite(&numpaths, sizeof(size_t), 1, f)!=1: raise IOError
    for k in range(numpaths):
      np = self.thisptr[0][k].size()
      IF DEBUG:   writeDebug("  WRITING PATH %d with %d points\n" % (k, np))
      if   io.fwrite(&np,       sizeof(size_t), 1, f)!=1: raise IOError
      for i in range(np):
        p = &self.thisptr[0][k][i]
        if io.fwrite(&p[0].X,   sizeof(double), 1, f)!=1: raise IOError
        if io.fwrite(&p[0].Y,   sizeof(double), 1, f)!=1: raise IOError
        IF DEBUG: writeDebug("    WRITING POINT %d: %f, %f\n" % (i, p[0].X, p[0].Y))
      
  cdef fromFileObject(self, io.FILE *f):
    """low level read function"""
    cdef size_t oldsize, numpaths, k, i, np
    cdef c.DoublePoint * p
    IF DEBUG:     writeDebug("    READING a clipperpath\n")
    if     io.fread(&numpaths, sizeof(size_t), 1, f)!=1: raise IOError
    IF DEBUG:     writeDebug("      NUMPATHS: %d\n" % numpaths)
    oldsize = self.thisptr[0].size()
    self.thisptr[0].resize(oldsize+numpaths)
    for k in range(oldsize, oldsize+numpaths):
      if   io.fread(&np,       sizeof(size_t), 1, f)!=1: raise IOError
      IF DEBUG:   writeDebug("      IN PATH %d, numpoints: %d\n" % (k, np))
      self.thisptr[0][k].resize(np)
    for k in range(oldsize, oldsize+numpaths):
      for i in range(np):
        p = &self.thisptr[0][k][i]
        if io.fread(&p[0].X,   sizeof(double), 1, f)!=1: raise IOError
        if io.fread(&p[0].Y,   sizeof(double), 1, f)!=1: raise IOError
        IF DEBUG: writeDebug("      POINT %d: %f, %f\n" % (i, p[0].X, p[0].Y))

  def toStream(self, stream):
    """write in binary mode. If stream is a string, it is the name of the file to
    write to. If it is None, data will be written to standard output. Otherwise,
    it must be a file object. The stream is changed to binary mode, if necessary"""
    cdef noisfile = not isinstance(stream, File)
    cdef File f
    if noisfile: f = File(stream, 'wb', True)
    else:        f = stream
    self.toFileObject(f.f)
    if noisfile: f.close()
    
  def fromStream(self, stream):
    """read in binary mode. If stream is a string, it is the name of the file to
    read from. If it is None, data will be read from standard output. Otherwise,
    it must be a file object. The stream is changed to binary mode, if necessary.
    New data from the stream is added to existing data (i.e. existing paths are
    not removed before adding new ones)"""
    cdef noisfile = not isinstance(stream, File)
    cdef File f
    if noisfile: f = File(stream, 'rb', False)
    else:        f = stream
    self.fromFileObject(f.f)
    if noisfile: f.close()
    
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

  def __cinit__(self, obj=None):
    cdef ClipperPaths other
    self.thisptr = new c.Paths()
    if not (obj is None):
      if isinstance(obj, ClipperPaths):
        other = obj
        self.thisptr[0] = other.thisptr[0]
      elif isinstance(obj, list):
        numpyList2ClipperPaths(obj, self)
      else:
        raise ValueError('Error: cannot initialize ClipperPaths with object of type %s' % str(type(obj)))
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
  
  @cython.boundscheck(False)
  def addPath(self, cnp.ndarray[cnp.int64_t, ndim=2] path):
    cdef cnp.int64_t k
    cdef size_t n = self.thisptr[0].size()
    cdef c.Path *cpath
    if path.shape[1]!=2:
      raise ValueError("the path must be an array with two columns!") 
    if path.shape[0]==0:
      raise ValueError("the path cannot be empty!!!!") 
    self.thisptr[0].resize(n+1)
    cpath = &(self.thisptr[0][n])
    cpath[0].resize(path.shape[0])
    for k in range(path.shape[0]):
      cpath[0][k].X = path[k,0]
      cpath[0][k].Y = path[k,1]
  
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
  
  cdef toFileObject(self, io.FILE *f):
    """low level write function"""
    cdef size_t numpaths = self.thisptr[0].size()
    cdef size_t k, i, np, bytesize
    cdef c.IntPoint * p
    IF DEBUG:     writeDebug("  WRITING A CLIPPERPATH WITH %d PATHS\n" % numpaths)
    if     io.fwrite(&numpaths, sizeof(size_t), 1, f)!=1: raise IOError
    for k in range(numpaths):
      np = self.thisptr[0][k].size()
      IF DEBUG:   writeDebug("  WRITING PATH %d with %d points\n" % (k, np))
      if   io.fwrite(&np,       sizeof(size_t), 1, f)!=1: raise IOError
    for k in range(numpaths):
      for i in range(self.thisptr[0][k].size()):
        p = &self.thisptr[0][k][i]
        if io.fwrite(&p[0].X,   sizeof(c.cInt), 1, f)!=1: raise IOError
        if io.fwrite(&p[0].Y,   sizeof(c.cInt), 1, f)!=1: raise IOError
        IF DEBUG: writeDebug("    WRITING POINT %d: %d, %d\n" % (i, p[0].X, p[0].Y))
      
  cdef fromFileObject(self, io.FILE *f):
    """low level read function"""
    cdef size_t oldsize, numpaths, k, i, np, bytesize
    cdef c.IntPoint * p
    IF DEBUG:     writeDebug("    READING a clipperpath\n")
    if     io.fread(&numpaths, sizeof(size_t), 1, f)!=1: raise IOError
    IF DEBUG:     writeDebug("      NUMPATHS: %d\n" % numpaths)
    oldsize = self.thisptr[0].size()
    self.thisptr[0].resize(oldsize+numpaths)
    for k in range(oldsize, oldsize+numpaths):
      if   io.fread(&np,       sizeof(size_t), 1, f)!=1: raise IOError
      IF DEBUG:   writeDebug("      IN PATH %d, numpoints: %d\n" % (k, np))
      self.thisptr[0][k].resize(np)
    for k in range(oldsize, oldsize+numpaths):
      for i in range(self.thisptr[0][k].size()):
        p = &self.thisptr[0][k][i]
        if io.fread(&p[0].X,   sizeof(c.cInt), 1, f)!=1: raise IOError
        if io.fread(&p[0].Y,   sizeof(c.cInt), 1, f)!=1: raise IOError
        IF DEBUG: writeDebug("      POINT %d: %d, %d\n" % (i, p[0].X, p[0].Y))

  def toStream(self, stream):
    """write in binary mode. If stream is a string, it is the name of the file to
    write to. If it is None, data will be written to standard output. Otherwise,
    it must be a file object. The stream is changed to binary mode, if necessary"""
    cdef noisfile = not isinstance(stream, File)
    cdef File f
    if noisfile: f = File(stream, 'wb', True)
    else:        f = stream
    self.toFileObject(f.f)
    if noisfile: f.close()
    
  def fromStream(self, stream):
    """read in binary mode. If stream is a string, it is the name of the file to
    read from. If it is None, data will be read from standard output. Otherwise,
    it must be a file object. The stream is changed to binary mode, if necessary.
    New data from the stream is added to existing data (i.e. existing paths are
    not removed before adding new ones)"""
    cdef noisfile = not isinstance(stream, File)
    cdef File f
    if noisfile: f = File(stream, 'rb', False)
    else:        f = stream
    self.fromFileObject(f.f)
    if noisfile: f.close()
    
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

@cython.boundscheck(False)
cpdef numpyList2ClipperPaths(list arrays, ClipperPaths paths=None):
  """convert a list of numpy arrays to a ClipperPaths object.
  WARNING: the arrays must still be bidimensional, with two columns, and of type int64, otherwise the conversion will fail!"""
  cdef cnp.npy_intp length = len(arrays)
  cdef c.Path * path
  cdef size_t npath, npoint, lenpoints
  cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] array
  cdef cnp.ndarray zs = cnp.PyArray_EMPTY(1, &length, cnp.NPY_FLOAT64, 0)
  if paths is None:
    paths = ClipperPaths()
  else:
    paths.thisptr[0].clear()
  paths.thisptr[0].resize(length)
  #for each layer
  for npath in range(<size_t>length):
    array                         = arrays[npath]
    lenpoints                     = array.shape[0]
    paths.thisptr[0][npath].resize(lenpoints)
    path                          = &paths.thisptr[0][npath]
    for npoint in range(lenpoints):
      path[0][npoint].X           = array[npoint,0]
      path[0][npoint].Y           = array[npoint,1]
  return paths


cdef PY3 = sys.version_info >= (3,0)
    
cdef class File:
  """Custom and (hopefully) fast lightweight wrapper for files.
  It works with strings representing file names, or stdin/stdout
  if None is provided (forcing them to binary mode)"""
  
  def __cinit__(self, object stream, str mode="wb", bool write=True):
    """open a FILE* object"""
    cdef fileno
    self.iswrite   = write
    if   isinstance(stream, basestring):
      self.doclose = True
      #CANNOT PORT DIRECTLY TO PYTHON 3: THIS STATEMET PRODUCES A SEGFAULT OR SOMETHING LIKE THAT IN WINPYTHON 3.4, BUT WORKS ON WINPYTHON 2.7.9
      self.f       = io.fopen(stream, mode)
      #if self.f==NULL:
      #  raise Exception("Could not open file "+stream)
    elif stream is None:
      self.doclose = False
      if write:
        stream     = sys.stdout #f = io.stdout
      else:
        stream     = sys.stdin  #f = io.stdin
      if PY3:
        stream = stream.buffer #python 3 ready
      fileno       = stream.fileno()
      if Windows:
        msvcrt.setmode(fileno, os.O_BINARY)
      self.f       = io.fdopen(fileno, mode)
      #if self.f==NULL:
      #  if write: raise Exception('Could not reopen stdout')
      #  else:     raise Exception('Could not reopen stdin')
    else:
      raise IOError("This class can only open new files or reopen stdin/stdout")

  def isValid(self):
    return self.f!=NULL
      
  cpdef close(self):
    """close the file just once, if it has to be done"""
    if self.doclose:
      io.fclose(self.f)
      self.doclose = False
  
  cpdef flush(self):
    """flush the buffers"""
    io.fflush(self.f)
  
  #it is redundant to expose these methods, but we need them upstream
  def readDouble(self):
    cdef double v
    if io.fread(&v, sizeof(v),  1, self.f)!=1: raise IOError
    return v  
  
  def readInt64(self):
    cdef cnp.int64_t v
    if io.fread(&v, sizeof(v),  1, self.f)!=1: raise IOError
    return v

  def readInt32(self):
    cdef cnp.int32_t v
    if io.fread(&v, sizeof(v),  1, self.f)!=1: raise IOError
    return v

  def writeInt32(self, int v):
    cdef cnp.int32_t v1
    v1 = v
    if io.fwrite(&v1, sizeof(v1), 1, self.f)!=1: raise IOError

  def write(self, object v):
    cdef cnp.int64_t v1
    cdef double      v2
    if   isinstance(v, int) or isinstance(v, long):
      v1 = v
      if io.fwrite(&v1, sizeof(v1), 1, self.f)!=1: raise IOError
    elif isinstance(v, float):
      v2 = v
      if io.fwrite(&v2, sizeof(v2), 1, self.f)!=1: raise IOError
    else:
      raise IOError
  
  def __dealloc__(self):
    """make sure it is closed"""
    self.close()

def read3DDoublePathsFromFile(File f):
  cdef cnp.npy_intp *dims  = [0,3]
  cdef cnp.int64_t numpaths
  cdef cnp.int64_t numpoints
  cdef cnp.int64_t i
  cdef cnp.int64_t numread
  cdef cnp.ndarray  points
  cdef list paths
  if io.fread(&numpaths, sizeof(numpaths),  1, f.f)!=1: raise IOError
  paths = [None]*numpaths
  for i in range(numpaths):
    if io.fread(&numpoints, sizeof(numpaths),  1, f.f)!=1: raise IOError
    dims[0] = numpoints
    numread = numpoints*3
    points = cnp.PyArray_EMPTY(2, dims, cnp.NPY_FLOAT64, 0)
    if io.fread(points.data, sizeof(cnp.float64_t),  numread, f.f)!=<cnp.uint64_t>numread: raise IOError
    paths[i] = points
  return paths
  
#strides have to be computed just once
cdef cnp.npy_intp *pointstrides = [sizeof(c.IntPoint),
                                   <cnp.uint8_t*>&(<c.IntPoint*>NULL).Y - 
                                   <cnp.uint8_t*>&(<c.IntPoint*>NULL).X]
cdef cnp.npy_intp *dpointstrides = [sizeof(c.DoublePoint),
                                   <cnp.uint8_t*>&(<c.DoublePoint*>NULL).Y - 
                                   <cnp.uint8_t*>&(<c.DoublePoint*>NULL).X]

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

cdef cnp.ndarray DPath2arrayView(ClipperDPaths parent, DPath *path):
  """Similar to Polygon2arrayI, but instead of allocating a full-blown array,
  the returned array is a view into the underlying data"""
  cdef void         *data  = &(path[0][0].X)
  cdef cnp.npy_intp *dims  = [path[0].size(),2]
  cdef cnp.ndarray  result = cnp.PyArray_New(np.ndarray, 2, dims, cnp.NPY_DOUBLE, dpointstrides,
                                             data, -1, NPY_ARRAY_CARRAY, <object>NULL)
  ##result.base is of type PyObject*, so no reference counting with this assignment
  result.base              = <ref.PyObject*>parent
  ref.Py_INCREF(parent) #so, make sure that "result" owns a reference to "parent"
  #ref.Py_INCREF(result)
  return result
  