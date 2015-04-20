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
#from libcpp cimport bool


cdef extern from "libslic3r/TriangleMesh.hpp" namespace "Slic3r" nogil:
    cdef cppclass TriangleMesh:
        TriangleMesh() except +
        void ReadSTLFile(char* input_file) except +
        void write_ascii(char* output_file) except +
        void write_binary(char* output_file) except +
        void repair() except +
        void WriteOBJFile(char* output_file) except +
        void scale(float factor)
        #void scale(const Pointf3 &versor);
        void translate(float x, float y, float z)
        void rotate_x(float angle)
        void rotate_y(float angle)
        void rotate_z(float angle)
        void flip_x()
        void flip_y()
        void flip_z()
        void align_to_origin()
        #void rotate(double angle, Point* center);
        #TriangleMeshPtrs split() const;
        #void merge(const TriangleMesh &mesh);
        
#    cdef enum FacetEdgeType: feNone, feTop, feBottom, feHorizontal
#
#    cdef cppclass IntersectionLine
#    
#    cdef cppclass TriangleMeshSlicer:
#        TriangleMesh* mesh
#        TriangleMeshSlicer(TriangleMesh* _mesh) except +
#        void slice(vector[float] &z, vector[Polygons]* layers)
#        void slice(vector[float] &z, vector[ExPolygons]* layers)
#        #void slice_facet(float slice_z, const stl_facet &facet, const int &facet_idx, const float &min_z, const float &max_z, std::vector<IntersectionLine>* lines) const;
#        #void cut(float z, TriangleMesh* upper, TriangleMesh* lower);
    
    
    
