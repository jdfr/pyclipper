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

import pyslic3r.slicing as p

print "A BRIEF INTRODUCTION TO pyslic3r.slicing"

print """
pyslic3r.slicing is a python wrapper for some of the
functionality of Slic3r, intended to make it easy to
inspect slices visually and in the command line, in
order to aid development of algorithms to process slices.

The main documentation for pyslic3r is the source code
itself, but a few examples are also provided.

Slic3r slices the data into a list of layers, where each
layer is list of ExPolygons, where each ExPolygon is a
contour (counter-clockwise closed path) with a number of
holes (clockwise closed path). The paths are just lists
of int64 points.

PLEASE NOTE: in 3D modelling, it is common to use floating
point to represent coordinates. However, slic3r uses
scaled int64 coordinates internally (converting to/from 
floating point as required by the workflow), because
some classes of geometric algorithms perform very poorly
with floating point coordinates (because of roundoff
errors). Consequently, pyslic3r also uses scaled int64 
coordinates, although there are easy ways to convert the
coordinates back to floating points. Slic3r's scaling
factor is hardcoded both in C++ and Perl. Un-hardcoding it
would very probably require a very extensive patch, so
it is also hardcoded in pyslic3r. Should you rather than un-hardcoding it in pyslic

Let's do some slicing!

SETUP"""

print """  'mesh' is just an in-memory representation of a STL file"
mesh   = p.TriangleMesh('test.stl')"""
mesh   = p.TriangleMesh('test.stl')

print """  'zs' is the list of Z values to slice the mesh"
zs     = n.array([0.1, 5.1])"""
zs     = n.array([0.1, 5.1])
#zs =  mesh.slicePlanes(0.1, 'constant')

print """  'sliced' is a wrapper for the C++ data structure that holds the result of the slicing"
sliced = mesh.doslice(zs)"""
sliced = mesh.doslice(zs)

print "  'sliced' is an object of type %s" % str(type(sliced))

print """
ACCESING TO THE DATA INSIDE SlicedModel:
"""

print "number of layers: len(sliced)==%d, also sliced.numLayers()==%d" % (len(sliced), sliced.numLayers())

print "number of ExPolygons in the first  layer: len(sliced[0])==%d, also sliced.numExPolygons(0)==%d" % (len(sliced[0]), sliced.numExPolygons(0))

print "z value of the first  layer: sliced[0, 'z']==%f" % sliced[0, 'z']

print "number of ExPolygons in the second layer: len(sliced[1])==%d" % len(sliced[1])

print "z value of the second layer: sliced[1, 'z']==%f" % sliced[1, 'z']

print "all z values: sliced.zs==%s" % str(sliced.zs)

print "second layer, first expolygon, coordinates of the contour: sliced[1,0,'c']"

print sliced[1,0,'c']

print "same as the previous one, but coordinates in original scale in floating point, instead of scaled int64: sliced.contour(1, 0, asInteger=False)"

print sliced.contour(1, 0, asInteger=False)

print "second layer, first expolygon, number of holes: len(sliced[1,0].holes)==%d, also sliced.numHoles(1, 0)==%d" % (len(sliced[1,0].holes), sliced.numHoles(1, 0))

print "second layer, first expolygon, coordinates of the first hole: sliced[1,0,0]"

print sliced[1,0,0]

print "same as the previous one, but coordinates in original scale in floating point, instead of scaled int64: sliced.hole(1, 0, 0, asInteger=False)"

print sliced.hole(1, 0, 0, asInteger=False)

print """
SlicedModel IS A THIN WRAPPER AROUND A Slic3r DATA STRUCTURE.
IT IS SOMEWHAT RIGID (NO PYTHONIC WAY TO MODIFY NUMBER OF
LAYERS OR REORDER LAYERS, FOR EXAMPLE):"""

print "type of sliced: %s" % str(type(sliced))

print """
WHEN ACCESSING DIRECTLY CONTOURS OR HOLES, THE RESULT
IS A VIEW ON THE ORIGINAL DATA, SO IT IS WRITABLE:
"""

print " old value: sliced[1,0,0][0,0]==%d" % sliced[1,0,0][0,0]

print "    assign: sliced[1,0,0][0,0]=12345"

sliced[1,0,0][0,0]=12345

print " new value: sliced[1,0,0][0,0]==%d" % sliced[1,0,0][0,0]

print ""

print " old value: sliced[1,0,'c'][0,0]==%d" % sliced[1,0,'c'][0,0]

print "    assign: sliced[1,0,'c'][0,0]=12345"

sliced[1,0,'c'][0,0]=12345

print " new value: sliced[1,0,'c'][0,0]==%d" % sliced[1,0,'c'][0,0]

print """
WHEN ACCESSING WITHOUT SPECIFYING INDEXES ALL THE WAY
TO CONTOURS OR HOLES, THE RESULTS ARE OTHER OBJECTS,
WHOSE DATA IS A COPY OF THE ORIGINAL. THESE OBJECTS
ARE FLEXIBLE (THEIR CONTENTS CAN BE CHANGED AT WILL
FROM PYTHON):
"""

print "type of sliced                 : %s" % str(type(sliced))

print "type of sliced.zs              : %s" % str(type(sliced.zs))

print "type of sliced[1]              : %s" % str(type(sliced[1]))

print "type of sliced[1].z            : %s" % str(type(sliced[1].z))

print "type of sliced[1].expolygons   : %s" % str(type(sliced[1].expolygons))

print "type of sliced[1].expolygons[0]: %s" % str(type(sliced[1].expolygons[0]))

print "type of sliced[1][0]           : %s" % str(type(sliced[1][0]))

print "type of sliced[1,0]            : %s" % str(type(sliced[1,0]))

print "type of sliced[1,0].contour    : %s" % str(type(sliced[1,0].contour))

print "type of sliced[1,0].holes      : %s" % str(type(sliced[1,0].holes))

print "type of sliced[1,0].holes[0]   : %s" % str(type(sliced[1,0].holes[0]))


print """
THERE IS ALSO A FLEXIBLE OBJECT EQUIVALENT TO THE WHOLE SlicedModel:
"""

print "type of sliced             : %s" % str(type(sliced))

print "sc = sliced.toSliceCollection()"
sc = sliced.toSliceCollection()

print "type of sc                : %s" % str(type(sliced.toSliceCollection()))

print "type of sc.slices         : %s" % str(type(sliced.toSliceCollection().slices))

print "type of sc.slices[0]      : %s" % str(type(sliced.toSliceCollection().slices[0]))

print "type of sc.toSlicedModel(): %s" % str(type(sliced.toSliceCollection().toSlicedModel()))

print """
SOME UTILITY METHODS SlicedModel:

model1.merge(model2): merges two SlicedModels assuming
that the the ExPolygons at identical z values do not
overlap

model.layersToClipperPaths(): returns an iterator yielding
each slice as a ClipperPaths object (more on that in ex02.py)

model.setLayerFromClipperObject(n, object): set the n-th
slice to the polygons specified in the clipper object
(more on that in ex02.py)

model.save(filename, mode): save the model to a file in
'ply' or 'svg' (slic3r-like) formats

model.copy(): return a copy of the object (the wrapped
C++ is deep-copied, the Z values are shallow-copied)

model.getBoundingBox(): return the overall bounding box
in the XY plane (for all slices combined)
"""
