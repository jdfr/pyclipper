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

from libcpp.vector cimport vector
#from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "clipper.hpp" namespace "ClipperLib" nogil:
  
  ############################################################################
  #ENUMS
  
  cdef enum ClipType:
    ctIntersection, ctUnion, ctDifference, ctXor
  cdef enum PolyType:
    ptSubject, ptClip
  cdef enum PolyFillType:
    pftEvenOdd, pftNonZero, pftPositive, pftNegative
  cdef enum InitOptions:
    ioReverseSolution = 1, ioStrictlySimple = 2, ioPreserveCollinear = 4
  cdef enum JoinType:
    jtSquare, jtRound, jtMiter
  cdef enum EndType:
    etClosedPolygon, etClosedLine, etOpenButt, etOpenSquare, etOpenRound
  
  ############################################################################
  #BASIC TYPES
  
  ctypedef signed long long cInt
  
  cdef struct DoublePoint:
    double X
    double Y
    DoublePoint(double, double)
  
  cdef struct IntPoint:
    cInt X
    cInt Y
    IntPoint(cInt, cInt)
  
  cdef struct IntRect:
    cInt left
    cInt top
    cInt right
    cInt bottom
    
  ctypedef vector[IntPoint] Path
  ctypedef vector[Path] Paths
  
  ############################################################################
  #BASIC FUNCTIONALITY

  bool Orientation(const Path &poly)
  double Area(const Path &poly)
  int PointInPolygon(const IntPoint &pt, const Path &path)
  
  void ReversePath(Path& p)
  void ReversePaths(Paths& p)
