from libcpp cimport bool

cimport numpy as cnp

cimport slic3r_defs as s3

cdef class SlicedModel:
  """wrapper for the Slic3r data structure for a list of sliced layers"""
  cdef s3.SLICEDMODEL *thisptr
  cdef cnp.ndarray zvalues
  cdef bool slicesAreOrdered(self)
  cdef void _removeLayers(self, unsigned int init, unsigned int end) nogil
  cdef void _removeExPolygons(self, unsigned int nlayer, unsigned int init, unsigned int end) nogil
  cdef void _removeHoles(self, unsigned int nlayer, unsigned int nexp, unsigned int init, unsigned int end) nogil
  cpdef unsigned int numLayers(self)
  cpdef unsigned int numExPolygons(self, unsigned int nlayer)
  cpdef unsigned int numHoles(self, unsigned int nlayer, unsigned int nExpolygon)
  cdef cnp.ndarray _contour(self, unsigned int nlayer, unsigned int nExpolygon, bool asInteger)
  cdef cnp.ndarray _hole(self, unsigned int nlayer, unsigned int nExpolygon, unsigned int nhole, bool asInteger)
