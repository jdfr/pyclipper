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
from libcpp         cimport bool

cimport numpy            as cnp
import  numpy            as  np

import os.path           as op

from numpy.math     cimport NAN, isnan

from numbers         import Number

#from libc.stdio cimport printf, puts

from    slic3r_defs cimport *

cimport SlicedModel
from    SlicedModel cimport SlicedModel

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
  cpdef cnp.ndarray[cnp.float64_t, ndim=2] boundingBox(self):
    cdef BoundingBoxf3 bb = self.thisptr[0].bounding_box()
    cdef cnp.npy_intp *retsize = [3,2]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] rbb = cnp.PyArray_EMPTY(2, retsize, cnp.NPY_FLOAT64, 0)
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
    cdef cnp.ndarray[cnp.float64_t, ndim=2] bb = self.boundingBox()
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
    cdef cnp.ndarray[cnp.float64_t, ndim=2] bb = self.boundingBox()
    cdef cnp.ndarray zs
    if mode[0]=='c': #'constant'
      if isnan(startshift):
        startshift = value
      zs = np.arange(bb[2,0]+startshift,bb[2,1], value, dtype=np.float64)
      if fabs(zs[-1]-bb[2,1])<np.spacing(bb[2,1]):
        zs = zs[:-1]
    elif mode[0]=='f':#'fixed'
      if isnan(startshift):
        zs = np.linspace(bb[2,0],bb[2,1], value+2)[1:-1]
      else:
        zs = np.linspace(bb[2,0]+startshift,bb[2,1], value, endpoint=False)
      zs = zs.astype(np.float64)
    else:
      raise ValueError('Invalid Slice Mode')
    return zs
    
  @cython.boundscheck(False)
  def doslice(self, inputs, double safety_offset=DEFAULT_SLICING_SAFETY_OFFSET):
    """generate a sliced model of this mesh"""
    cdef size_t k, sz
    cdef vector[cnp.float32_t] zsv
    cdef cnp.ndarray zs
    #we cannot allocate the object in the stack because cython requires it to have a contructor without args
    cdef TriangleMeshSlicer *slicer = new TriangleMeshSlicer(self.thisptr, safety_offset)
    cdef SlicedModel layers
    try:
      if isinstance(inputs, np.ndarray):
        zs       = inputs
        sz       = zs.size
        zsv.resize(sz)
        for k in range(sz):
          zsv[k] = zs[k]
        layers   = SlicedModel(zs.astype(np.float64))
      elif isinstance(inputs, Number):
        zsv.resize(1)
        zsv[0]   = inputs
        layers   = SlicedModel(np.array(inputs, dtype=np.float64))
      else:
        raise ValueError('Invalid specification for z values')
      with nogil:
        slicer.slice(zsv, layers.thisptr)
      return layers
    finally:
      del slicer