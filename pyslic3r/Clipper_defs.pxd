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
  
  cdef cppclass PolyNode  
  ctypedef vector[PolyNode*] PolyNodes
  
  cdef cppclass PolyNode:
    Path Contour
    PolyNodes Childs
    PolyNode* Parent
    PolyNode* GetNext() const
    bool IsHole() const
    bool IsOpen() const
    int ChildCount() const
  
  cdef cppclass PolyTree(PolyNode):
    PolyNode* GetFirst() const
    void Clear()
    int Total() const

  ############################################################################
  #BASIC FUNCTIONALITY

  bool Orientation(const Path &poly)
  double Area(const Path &poly)
  int PointInPolygon(const IntPoint &pt, const Path &path)
  
  void SimplifyPolygon (const Path  &in_poly,  Paths &out_polys, PolyFillType fillType)
  void SimplifyPolygons(const Paths &in_polys, Paths &out_polys, PolyFillType fillType)
  void SimplifyPolygons(Paths &polys, PolyFillType fillType)
  
  void CleanPolygon(const Path& in_poly, Path& out_poly, double distance)
  void CleanPolygon(const Path& in_poly, Path& out_poly)
  void CleanPolygon(Path& poly, double distance);
  void CleanPolygon(Path& poly)
  void CleanPolygons(const Paths& in_polys, Paths& out_polys, double distance)
  void CleanPolygons(const Paths& in_polys, Paths& out_polys)
  void CleanPolygons(Paths& polys, double distance);
  void CleanPolygons(Paths& polys);
  
  void MinkowskiSum(const Path& pattern, const Path& path, Paths& solution, bool pathIsClosed)
  void MinkowskiSum(const Path& pattern, const Paths& paths, Paths& solution, bool pathIsClosed)
  void MinkowskiDiff(const Path& poly1, const Path& poly2, Paths& solution)
  
  void PolyTreeToPaths(const PolyTree& polytree, Paths& paths)
  void ClosedPathsFromPolyTree(const PolyTree& polytree, Paths& paths)
  void OpenPathsFromPolyTree(PolyTree& polytree, Paths& paths)
  
  void ReversePath(Path& p)
  void ReversePaths(Paths& p)

  ############################################################################
  #CLIPPING & OFFSETING

  cdef cppclass Clipper:
    Clipper(int initOptions)
    Clipper()
    bool AddPath (const Path  &pg,  PolyType PolyTyp, bool Closed) except +
    bool AddPaths(const Paths &ppg, PolyType PolyTyp, bool Closed) except +
    bool Execute(ClipType clipType, Paths    &solution, PolyFillType fillType) except +
    bool Execute(ClipType clipType, Paths    &solution, PolyFillType subjFillType, PolyFillType clipFillType) except +
    bool Execute(ClipType clipType, PolyTree &polytree, PolyFillType fillType) except +
    bool Execute(ClipType clipType, PolyTree &polytree, PolyFillType subjFillType, PolyFillType clipFillType) except +
    void Clear() except +
    IntRect GetBounds()
    bool ReverseSolution()
    void ReverseSolution(bool value)
    bool StrictlySimple()
    void StrictlySimple(bool value)
    bool PreserveCollinear()
    void PreserveCollinear(bool value)

  cdef cppclass ClipperOffset:
    ClipperOffset(double miterLimit, double roundPrecision)
    ClipperOffset(double miterLimit)
    ClipperOffset()
    void AddPath (const Path & path,  JoinType joinType, EndType endType)
    void AddPaths(const Paths& paths, JoinType joinType, EndType endType)
    void Execute(Paths& solution, double delta) except +
    void Execute(PolyTree& solution, double delta) except +
    void Clear() except +
    double MiterLimit
    double ArcTolerance
