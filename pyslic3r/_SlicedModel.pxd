from libcpp cimport bool

cimport numpy as cnp

cimport slic3r_defs as s3

cimport  Clipper as  c
#cimport _Clipper as _c

cdef class SlicedModel:
  """wrapper for the Slic3r data structure for a list of sliced layers"""
  cdef  s3.SLICEDMODEL *thisptr
  cdef  cnp.ndarray zvalues
  cdef  bool slicesAreOrdered     (self)
  cdef  void _removeLayers        (self,                                         size_t init, size_t end) nogil
  cdef  void _removeExPolygons    (self, size_t nlayer,                    size_t init, size_t end) nogil
  cdef  void _removeHoles         (self, size_t nlayer, size_t nexp, size_t init, size_t end) nogil
  cpdef size_t numLayers          (self)
  cpdef size_t numExPolygons      (self, size_t nlayer)
  cpdef size_t numHoles           (self, size_t nlayer, size_t nexp)
  cpdef object  toLayerList       (self,                                         bool asInteger=?, bool asView=?, rang=?)
  cpdef object  toExPolygonList   (self, size_t nlayer,                    bool asInteger=?, bool asView=?, rang=?)
  cdef  object _toExPolygonList   (self, size_t nlayer,                    bool asInteger=?, bool asView=?, rang=?)
  cpdef object  toHoleList        (self, size_t nlayer, size_t nexp, bool asInteger=?, bool asView=?, rang=?)
  cdef  object _toHoleList        (self, size_t nlayer, size_t nexp, bool asInteger=?, bool asView=?, rang=?)
  cdef  cnp.ndarray Polygon2array (self, s3.Polygon *pol,                        bool asInteger=?, bool asView=?)
  cdef  c.Paths* _layerToClipperPaths        (self, size_t nlayer, c.Paths    *output) nogil
  cdef  void     _setLayerFromClipperPaths   (self, size_t nlayer, c.Paths    *inputs) nogil
  cdef  void     _setLayerFromClipperPolyTree(self, size_t nlayer, c.PolyTree *inputs) nogil
