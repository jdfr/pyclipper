from libcpp cimport bool

cimport numpy as cnp

cimport slic3r_defs as s3

cdef class SlicedModel:
  """wrapper for the Slic3r data structure for a list of sliced layers"""
  cdef  s3.SLICEDMODEL *thisptr
  cdef  cnp.ndarray zvalues
  cdef  bool slicesAreOrdered     (self)
  cdef  void _removeLayers        (self,                                         unsigned int init, unsigned int end) nogil
  cdef  void _removeExPolygons    (self, unsigned int nlayer,                    unsigned int init, unsigned int end) nogil
  cdef  void _removeHoles         (self, unsigned int nlayer, unsigned int nexp, unsigned int init, unsigned int end) nogil
  cpdef unsigned int numLayers    (self)
  cpdef unsigned int numExPolygons(self, unsigned int nlayer)
  cpdef unsigned int numHoles     (self, unsigned int nlayer, unsigned int nexp)
  cpdef object  toLayerList       (self,                                         bool asInteger=?, bool asView=?, rang=?)
  cpdef object  toExPolygonList   (self, unsigned int nlayer,                    bool asInteger=?, bool asView=?, rang=?)
  cdef  object _toExPolygonList   (self, unsigned int nlayer,                    bool asInteger=?, bool asView=?, rang=?)
  cpdef object  toHoleList        (self, unsigned int nlayer, unsigned int nexp, bool asInteger=?, bool asView=?, rang=?)
  cdef  object _toHoleList        (self, unsigned int nlayer, unsigned int nexp, bool asInteger=?, bool asView=?, rang=?)
  cdef  cnp.ndarray Polygon2array (self, s3.Polygon *pol,                        bool asInteger=?, bool asView=?)



