#cython: embedsignature=True

cimport cython
from libcpp cimport bool

cimport numpy as cnp
import numpy as np

import os.path as op

from numpy.math cimport NAN, isnan

from slic3r_defs cimport *

cimport _SlicedModel
from _SlicedModel cimport SlicedModel

#np.import_array()
cnp.import_array()

cdef extern from "math.h" nogil:
  double fabs(double)

#######################################################################
########## TriangleMesh WRAPPER CLASS ##########
#######################################################################

cdef class TriangleMesh:
  """class to represent a STL mesh and slice it"""
  
  def __cinit__(self, filename):
    self.thisptr = new _TriangleMesh()
    if not op.isfile(filename):
      raise Exception('This file does not exist: '+filename)
    try:
      self.thisptr.ReadSTLFile(filename)
    except:
      raise Exception('Could not import the STL file '+filename)
  def __dealloc__(self):
    del self.thisptr
  
  @cython.boundscheck(False)  
  def translate(self, double x,double y,double z):
    self.thisptr[0].translate(x,y,z)
  
  @cython.boundscheck(False)  
  def numTriangles(self):
    return self.thisptr[0].facets_count()
  
  @cython.boundscheck(False)  
  def repairNeeded(self):
    return self.thisptr[0].needed_repair()
  
  @cython.boundscheck(False)  
  def hasBeenRepaired(self):
    return self.thisptr[0].repaired
  
  @cython.boundscheck(False)  
  def repair(self):
    self.thisptr[0].repair()
  
  @cython.boundscheck(False)  
  cpdef cnp.ndarray[cnp.float32_t, ndim=2] boundingBox(self):
    cdef BoundingBoxf3 bb = self.thisptr[0].bounding_box()
    cdef cnp.npy_intp *retsize = [3,2]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] rbb = cnp.PyArray_EMPTY(2, retsize, cnp.NPY_FLOAT32, 0)
    rbb[0,0] = bb.min.x
    rbb[1,0] = bb.min.y
    rbb[2,0] = bb.min.z
    rbb[0,1] = bb.max.x
    rbb[1,1] = bb.max.y
    rbb[2,1] = bb.max.z
    return rbb
  
  @cython.boundscheck(False)  
  def alignToCenter(self):
    "center in the plane XY, and rebase in Z so the model is in touching the XY plane, in the +Z half-space"
    cdef cnp.ndarray[cnp.float32_t, ndim=2] bb = self.boundingBox()
    cdef cnp.ndarray c  = np.sum(bb, axis=1)/2
    self.thisptr[0].translate(-c[0], -c[1], -bb[2,0])
  
  def add(self, TriangleMesh other):
    """add another mesh to this one. WARNING: no boolean operation is performed.
    If the meshes intersect, unexpected errors will likely follow"""
    self.thisptr[0].merge(other.thisptr[0])
  
  def save(self, basestring filename, basestring mode='stlb'):
    cdef char *FILENAME = filename
    model = mode.lower()
    if mode=='stl':
      mode = 'stlb'
    if mode=='stlb':
      with nogil:
        self.thisptr[0].write_binary(FILENAME)
    elif mode=='stla':
      with nogil:
        self.thisptr[0].write_ascii(FILENAME)
    elif mode=='obj':
      with nogil:
        self.thisptr[0].WriteOBJFile(FILENAME)
    else:
      raise Exception('mode not understood: '+mode)

  @cython.boundscheck(False)
  def slicePlanes(self, double value, basestring mode='c', double startshift=NAN):
    """Simple slicing, with the parameter "mode" specifying either 'constant' z steps, or a 'fixed' number of steps.
    if specified, the parameter startshift is the height of the first slice"""
    cdef cnp.ndarray[cnp.float32_t, ndim=2] bb = self.boundingBox()
    cdef cnp.ndarray zs
    if mode[0]=='c': #'constant'
      if isnan(startshift):
        startshift = value
      zs = np.arange(bb[2,0]+startshift,bb[2,1], value, dtype=np.float32)
      if fabs(zs[-1]-bb[2,1])<np.spacing(bb[2,1]):
        zs = zs[:-1]
    elif mode[0]=='f':#'fixed'
      if isnan(startshift):
        zs = np.linspace(bb[2,0],bb[2,1], value+2)[1:-1]
      else:
        zs = np.linspace(bb[2,0]+startshift,bb[2,1], value, endpoint=False)
      zs = zs.astype(np.float32)
    else:
      raise ValueError('Invalid Slice Mode')
    return zs
    
  @cython.boundscheck(False)
  def doslice(self, cnp.ndarray[cnp.float32_t, ndim=1] zs, double safety_offset=DEFAULT_SLICING_SAFETY_OFFSET):
    """generate a sliced model of this mesh"""
    cdef int k, k1, sz, sz1
    cdef vector[float] zsv
    #we cannot allocate the object in the stack because cython requires it to have a contructor without args
    cdef TriangleMeshSlicer *slicer = new TriangleMeshSlicer(self.thisptr, DEFAULT_SLICING_SAFETY_OFFSET)
    cdef SlicedModel layers = SlicedModel(zs.copy())
    try:
      sz = zs.size
      zsv.resize(sz)
      for k in range(sz):
        zsv[k] = zs[k]
      with nogil:
        slicer.slice(zsv, layers.thisptr)
      return layers
    finally:
      del slicer