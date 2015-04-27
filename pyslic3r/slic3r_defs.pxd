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


#from libc.stddef cimport size_t

from libcpp.vector cimport vector
#from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "libslic3r/libslic3r.h" nogil:
  ctypedef long coord_t
  ctypedef double coordf_t
  cdef double SCALING_FACTOR
  cdef double unscale(coord_t)
  cdef double scale(double)
  
  
cdef extern from "libslic3r/Point.hpp" namespace "Slic3r" nogil:
  cdef cppclass Point:
    coord_t x
    coord_t y
  ctypedef vector[Point] Points
  
  cdef cppclass Pointf3:
    coordf_t x
    coordf_t y
    coordf_t z


cdef extern from "libslic3r/BoundingBox.hpp" namespace "Slic3r" nogil:
  cdef cppclass BoundingBoxf3:
    Pointf3 min
    Pointf3 max


cdef extern from "libslic3r/MultiPoint.hpp" namespace "Slic3r" nogil:
  cdef cppclass MultiPoint:
    Points points

cdef extern from "libslic3r/Polygon.hpp" namespace "Slic3r" nogil:
  cdef cppclass Polygon
  ctypedef vector[Polygon] Polygons
  cdef cppclass Polygon(MultiPoint):
    double area() const
    bool is_counter_clockwise() const
    bool is_clockwise() const
    bool make_counter_clockwise()
    bool make_clockwise()
    bool is_valid() const
    bool contains(const Point &point) const
    Polygons simplify(double tolerance) const
    void simplify(double tolerance, Polygons &polygons) const

cdef extern from "libslic3r/ExPolygon.hpp" namespace "Slic3r" nogil:
  cdef cppclass _ExPolygon "Slic3r::ExPolygon"
  ctypedef vector[_ExPolygon] ExPolygons
  cdef cppclass _ExPolygon "Slic3r::ExPolygon":
    Polygon contour
    Polygons holes
    double area() const
    bool is_valid() const
    bool contains(const Point &point) const
    bool has_boundary_point(const Point &point) const
    Polygons simplify_p(double tolerance) const
    ExPolygons simplify(double tolerance) const
    void simplify(double tolerance, ExPolygons &expolygons) const
    #void medial_axis(double max_width, double min_width, Polylines* polylines) const
    void triangulate(Polygons* polygons) const
    void triangulate_pp(Polygons* polygons) const
    void triangulate_p2t(Polygons* polygons) const

ctypedef vector[ExPolygons] SLICEDMODEL

cdef extern from "libslic3r/TriangleMesh.hpp" namespace "Slic3r" nogil:
  cdef cppclass _TriangleMesh "Slic3r::TriangleMesh":
    _TriangleMesh() except +
    void ReadSTLFile(char* input_file) except +
    void write_ascii(char* output_file) except +
    void write_binary(char* output_file) except +
    void repair() except +
    void WriteOBJFile(char* output_file) except +
    void scale(float factor)
    #void scale(const Pointf3 &versor)
    void translate(float x, float y, float z)
    void rotate_x(float angle)
    void rotate_y(float angle)
    void rotate_z(float angle)
    void flip_x()
    void flip_y()
    void flip_z()
    BoundingBoxf3 bounding_box() 
    void merge(const _TriangleMesh &mesh) except +
    bool repaired
    bool needed_repair() const
    size_t facets_count() const
    
        
  cdef double DEFAULT_SLICING_SAFETY_OFFSET "DEFAULT_SLICING_SAFETY_OFFSET"
    
  cdef cppclass TriangleMeshSlicer:
    _TriangleMesh* mesh
    TriangleMeshSlicer(_TriangleMesh* _mesh) except +
    TriangleMeshSlicer(_TriangleMesh* _mesh, double safety_offset) except +
    void slice(vector[float] &z, vector[Polygons]* layers) except +
    void slice(vector[float] &z, SLICEDMODEL *layers) except +
    #void slice_facet(float slice_z, const stl_facet &facet, const int &facet_idx, const float &min_z, const float &max_z, std::vector<IntersectionLine>* lines) const
    #void cut(float z, _TriangleMesh* upper, _TriangleMesh* lower)
    
    

cdef extern from "clipper.hpp" namespace "ClipperLib" nogil:
  
  ctypedef signed long long cInt
  
  cdef struct IntPoint:
    cInt X
    cInt Y
    
  ctypedef vector[IntPoint] Path
  ctypedef vector[Path] Paths
  
  cdef enum JoinType:
    jtSquare, jtRound, jtMiter
    
cdef extern from "libslic3r/ClipperUtils.hpp" namespace "Slic3r" nogil:
#NOTE: MOST OF THE FUNCTIONS DECLARED HERE HAVE DEFAULT ARGUMENTS, WHICH HAPPEN
#TO COINCIDE WITH THE DEFAULT ARGUMENTS DECLARED IN THE *.xsp PERL BINDINGS,
#AT LEAST IN THE VERSION OF SLIC3R THAT WE ARE USING

  void Slic3rMultiPoint_to_ClipperPath(const MultiPoint &inputt, Path* output)
  void Slic3rMultiPoints_to_ClipperPaths[T](const T &inputt, Paths* output)
  void ClipperPath_to_Slic3rMultiPoint[T](const Path &inputt, T* output, bool eraseOutput)
  void ClipperPaths_to_Slic3rMultiPoints[T](const Paths &inputt, T* output, bool eraseOutput)
  void ClipperPaths_to_Slic3rExPolygons(const Paths &input, ExPolygons* output, bool eraseOutput)
  void Add_Slic3rExPolygon_to_ClipperPaths(const _ExPolygon &inputt, Paths* output)
  void Slic3rExPolygons_to_ClipperPaths(const ExPolygons &inputt, Paths* output)
  
  void offset(const Polygons &polygons, ExPolygons* retval, const float delta) except + #version with implicit paramenters
  void offset(const Polygons &polygons, ExPolygons* retval, const float delta, double scale, JoinType joinType, double miterLimit, bool eraseOutput) except + 
  void offset(const Polygons &polygons,   Polygons* retval, const float delta) except + #version with implicit paramenters
  void offset(const Polygons &polygons,   Polygons* retval, const float delta, double scale, JoinType joinType, double miterLimit, bool eraseOutput) except + 

  void offset2(const _ExPolygon  &polygons, ExPolygons* retval, const float delta1, const float delta2, double scale, JoinType joinType, double miterLimit, bool eraseOutput) except + 
  void offset2(const _ExPolygon  &polygons,   Polygons* retval, const float delta1, const float delta2, double scale, JoinType joinType, double miterLimit, bool eraseOutput) except + 
#  void offset2(const ExPolygons &polygons, ExPolygons* retval, const float delta1, const float delta2, double scale, JoinType joinType, double miterLimit, bool eraseOutput) except + 
#  void offset2(const   Polygons &polygons, ExPolygons* retval, const float delta1, const float delta2, double scale, JoinType joinType, double miterLimit, bool eraseOutput) except + 
#  void offset2(const   Polygons &polygons,   Polygons* retval, const float delta1, const float delta2, double scale, JoinType joinType, double miterLimit, bool eraseOutput) except + 

  void diff [SubjectType, ResultType](const SubjectType &subject, const ExPolygons &clip, ResultType* retval, bool safety_offset_, bool eraseOutput) except + 
  void diff [SubjectType, ResultType](const SubjectType &subject, const   Polygons &clip, ResultType* retval, bool safety_offset_, bool eraseOutput) except + 
  
cdef extern from "glue.hpp" nogil:
  void mydiff(const ExPolygons &subject, const Polygons &clip, ExPolygons* retval, bool safety_offset_, bool eraseOutput) except +  #version with implicit paramenters

