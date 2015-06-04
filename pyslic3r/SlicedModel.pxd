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

from libcpp cimport bool

cimport numpy        as cnp

cimport  slic3r_defs as s3

cimport Clipper_defs as  c

cdef class SlicedModel:
  """wrapper for the Slic3r data structure for a list of sliced layers"""
  cdef  s3.SLICEDMODEL *thisptr
  cdef  cnp.ndarray zvalues
  cdef  bool slicesAreOrdered     (self)
  cdef  void _removeLayers        (self,                             size_t init, size_t end) nogil
  cdef  void _removeExPolygons    (self, size_t nlayer,              size_t init, size_t end) nogil
  cdef  void _removeHoles         (self, size_t nlayer, size_t nexp, size_t init, size_t end) nogil
  cpdef size_t numLayers          (self)
  cpdef size_t numExPolygons      (self, size_t nlayer)
  cpdef size_t numHoles           (self, size_t nlayer, size_t nexp)
  cpdef object  toLayerList       (self,                             bool asInteger=?, bool asView=?, rang=?)
  cpdef object  toExPolygonList   (self, size_t nlayer,              bool asInteger=?, bool asView=?, rang=?)
  cdef  object _toExPolygonList   (self, size_t nlayer,              bool asInteger=?, bool asView=?, rang=?)
  cpdef object  toHoleList        (self, size_t nlayer, size_t nexp, bool asInteger=?, bool asView=?, rang=?)
  cdef  object _toHoleList        (self, size_t nlayer, size_t nexp, bool asInteger=?, bool asView=?, rang=?)
  cdef  cnp.ndarray Polygon2array (self, s3.Polygon *pol,            bool asInteger=?, bool asView=?)
  cdef  c.Paths* _layerToClipperPaths        (self, size_t nlayer, c.Paths    *output) nogil
  cdef  void     _setLayerFromClipperPaths   (self, size_t nlayer, c.Paths    *inputs) nogil
  cdef  void     _setLayerFromClipperPolyTree(self, size_t nlayer, c.PolyTree *inputs) nogil
