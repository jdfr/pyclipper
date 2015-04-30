from libcpp cimport bool

cimport numpy as cnp

cimport slic3r_defs as s3

cdef class TriangleMesh:
  """class to represent a STL mesh and slice it"""
  cdef s3._TriangleMesh *thisptr
  cpdef cnp.ndarray[cnp.float64_t, ndim=2] boundingBox(self)
