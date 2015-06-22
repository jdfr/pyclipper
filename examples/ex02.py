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

import sys
import os.path as op
sys.path.append(op.abspath('..')) #this is to have pyslic3r in our PYTHONPATH

import numpy as n

import pyslic3r.Clipper as c
import pyslic3r.slicing as p

print "A BRIEF INTRODUCTION TO pyslic3r.Clipper"

print """
pyslic3r.Clipper is a python wrapper for most of the
functionality of Angus Johnson's ClipperLib, the 
library used by Slic3r to implement boolean & offseting
primitives in 2D polygons. It is a relatively low
level, thin wrapper, meaning that it does not intend
to make easy the use of the library, but it just
exposes ClipperLib's functionality to python, using it
to make it easier to develop clipping & offseting
workflows.

The main documentation for pyslic3r is the source code
itself, but a few examples are also provided.

ClipperLib's main data structure is ClipperLib::Paths,
a vector of vectors of 2D points in int64 coordinates,
which is wrapped in pyslic3r.Clipper by the class
pyslic3r.Clipper.ClipperPaths. It corresponds 1:1
to Slic3r::Polygons, and each 2D slice in a
SlicedModel (see ex01.py) can be converted from/to a
ClipperPaths.

SETUP
"""

print "  'mesh' is just an in-memory representation of a STL file"
mesh   = p.TriangleMesh('test.stl')
print "  'zs' is the list of Z values to slice the mesh"
zs     = n.array([0.1, 5.1])
#zs =  mesh.slicePlanes(0.1, 'constant')
print "  'sliced' is a wrapper for the C++ data structure that holds the result of the slicing"
sliced = mesh.doslice(zs)
print """  'pathss' is a list of ClipperPaths, each ClipperPaths
being a wrapper for the main C++ data structure used by ClipperLib"""
pathss = list(sliced.layersToClipperPaths())

print "  'pathss' is a list of ClipperPaths objects: %s" % str(map(type, pathss))

print "  let's use just the first ClipperPaths: paths = pathss[0]"
paths = pathss[0]

print """
ACCESING TO THE DATA INSIDE ClipperPaths:
"""

print "number of paths: len(paths)==%d" % len(paths)

print "coordinates of the first path:"

print paths[0]

print "reverse the first path, then show the coordinates again:"

paths.reverse(0)
print paths[0]

print "\nWHEN ACCESSING DIRECTLY THE PATHS, THE RESULT IS A VIEW ON THE ORIGINAL DATA, SO IT IS WRITABLE:"

old = paths[0][0,0]

print " old value: paths[0][0,0]==%d" % old

print "    assign: paths[0][0,0]=12345"

paths[0][0,0]=12345

print " new value: sliced[1,0,0][0,0]==%d" % paths[0][0,0]

paths[0][0,0] = old #restore the previous value


print """
paths are interpreted as closed, and can be
either contours or holes, depending on their orientation
(counter-clockwise for contours, clockwise for holes)."""

print "\norientations of the paths (true for counter-clockwise): paths.orientations()==%s" % str(paths.orientations())

print "\nareas of the paths (signed by the orientation): paths.areas()==%s" % str(paths.areas())

print "\nCreate an empty ClipperPaths: newp = ClipperPaths()"
newp = c.ClipperPaths()

print "\nadd a path: newp.addPath(array([(0, 0), (0, 10), (10, 10), (10, 0)], dtype=int64))"
newp.addPath(n.array([(0, 0), (0, 10), (10, 10), (10, 0)], dtype=n.int64))

print "show the newly added path: newp[0]"

print newp[0]

print """
ClipperPaths, and sequences of ClipperPaths, can be read/written from/to files
in a simple, compact and binary format. While this functionality is somewhat
redundant with SlicedModel's serialization capabilities, it is provided in this
module to do it in fast way for some use cases that require it. You can inspect
the source code to see all the IO facilities."""

print """
write a ClipperPaths to a file:
   newp.toStream('saved.paths')
read a ClipperPaths from a file:
   newp2 = ClipperPaths()
   newp2.fromStream('saved.paths')
   print newp2[0]"""

newp.toStream('saved.paths')
newp2 = c.ClipperPaths()
newp2.fromStream('saved.paths')
print newp2[0]

print "\nClipperPaths.pointInPolygon(n, x, y): test the position of a point relative to the n-th path: inside (1) / outside (0) / just in the path (-1)"

print "newp2.pointInPolygon(0, -1, -1)==%s" % str(newp2.pointInPolygon(0, -1, -1))
print "newp2.pointInPolygon(0,  0,  5)==%s" % str(newp2.pointInPolygon(0,  0,  5))
print "newp2.pointInPolygon(0,  5,  5)==%s" % str(newp2.pointInPolygon(0,  5,  5))

print """
CLIPPING & OFFSETING

ClipperClip and ClipperOffset are thin wrappers around
ClipperLib::Clipper and ClipperLib::ClipperOffset,
respectively. They are used in exactly the same way as
their ClipperLib counterparts. Please inspect the source code
and ex05.py in order to understand how to use them.
"""
