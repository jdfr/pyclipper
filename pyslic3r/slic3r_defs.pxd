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
  cdef cppclass ExPolygon
  ctypedef vector[ExPolygon] ExPolygons
  cdef cppclass ExPolygon:
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

cdef extern from "libslic3r/TriangleMesh.hpp" namespace "Slic3r" nogil:
  cdef cppclass TriangleMesh:
    TriangleMesh() except +
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
    void align_to_origin()
    #void rotate(double angle, Point* center)
    #TriangleMeshPtrs split() const
    void merge(const TriangleMesh &mesh)
        
    
  cdef cppclass TriangleMeshSlicer:
    TriangleMesh* mesh
    TriangleMeshSlicer(TriangleMesh* _mesh) except +
    void slice(vector[float] &z, vector[Polygons]* layers) except +
    void slice(vector[float] &z, vector[ExPolygons]* layers) except +
    #void slice_facet(float slice_z, const stl_facet &facet, const int &facet_idx, const float &min_z, const float &max_z, std::vector<IntersectionLine>* lines) const
    #void cut(float z, TriangleMesh* upper, TriangleMesh* lower)
    

cdef extern from "clipper.hpp" namespace "ClipperLib" nogil:
  cdef enum JoinType:
    jtSquare, jtRound, jtMiter
    
cdef extern from "libslic3r/ClipperUtils.hpp" namespace "Slic3r" nogil:
  #Let's start with the functions operating on ExPolygons, we will declare the others later if we need them  
  
#  void offset(const Polygons &polygons, ExPolygons* retval, const float delta)
#  void offset(const Polygons &polygons, ExPolygons* retval, const float delta, double scale)
#  void offset(const Polygons &polygons, ExPolygons* retval, const float delta, double scale, JoinType joinType)
  void offset(const Polygons &polygons, ExPolygons* retval, const float delta, double scale, JoinType joinType, double miterLimit)

#  void offset2(const Polygons &polygons, ExPolygons* retval, const float delta1, const float delta2)
#  void offset2(const Polygons &polygons, ExPolygons* retval, const float delta1, const float delta2, double scale)
#  void offset2(const Polygons &polygons, ExPolygons* retval, const float delta1, const float delta2, double scale, JoinType joinType)
  void offset2(const Polygons &polygons, ExPolygons* retval, const float delta1, const float delta2, double scale, JoinType joinType, double miterLimit)

#  void diff[SubjectType, ResultType](const SubjectType &subject, const ExPolygons &clip, ResultType* retval)
  void diff[SubjectType, ResultType](const SubjectType &subject, const ExPolygons &clip, ResultType* retval, bool safety_offset_)

