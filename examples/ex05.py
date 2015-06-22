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
import pyslic3r.plot2d  as p2

print """
SIMPLE EXAMPLE SLICE MODIFICATION WITH pyslic3r

As stated in the root directory's README, pyslic3r
is intended to enable scripted modifications to
slices, before they pass down the pipeline. While
these modifications can be done by hand, this can be
quite difficult in practice. Instead, modifications
are more likely to be performed by clipping & offseting
operations.

This file contains a very simple example of the kind
of modifications which are enabled by pyslic3r: we modify
an object to put smooth corners and add a roughly conical
hole. Of course, this particular kind of modification should
really be done to the 3D model rather than the slices. This
is just a simple example. But you get the idea: you can
make the slices smaller, remove detail, add or remove
things, or do different things to different slices, or
different parts of a slice.
"""

#some parameters
show3D          = True
show2D          = False
layerHeight     = 0.3
bottomNumLayers = 10

mesh            = p.TriangleMesh('cube.stl')
#Z coordinates are in the 3D model's original scale (in floating point)
zs              = mesh.slicePlanes(layerHeight, 'constant')
#the slices are in int64 coordinates (scaled by 1e6)
sliced          = mesh.doslice(zs)

#constants to be used in slices have to be scaled by 1e6
vaseThickness   = n.linspace(4*1e6, 2*1e6, sliced.numLayers()-bottomNumLayers)
openingRadius1  =     1*1e6
openingRadius2  =   0.5*1e6
shiftX          =    12*1e6
arcScale        = 0.001*1e6

#build the clipping and offseting objects, as well as auxiliary
#containers.ClipperOffset corresponds to ClipperLib::ClipperOffset
#and ClipperClip to ClipperLib::Clipper, and ClipperPaths to 
#ClipperLib::Paths. The parameters in the constructors correspond
#to parameters of the respective engines, which can also be 
#accessed and modified as properties of the respective objects.

offset          = c.ClipperOffset(joinType=c.jtRound,
                                  endType=c.etClosedPolygon,
                                  arcTolerance=arcScale)
clipper         = c.ClipperClip(clipType=c.ctDifference)

opened          = c.ClipperPaths()
aux1            = c.ClipperPaths()
aux2            = c.ClipperPaths()
aux3            = c.ClipperPolyTree()

#make an empty copy of the slices (same number of slices,
#same Z levels, but all slices are empty)
newsliced       = sliced.copyEmpty()

#do the modifications
for nlayer, paths in enumerate(sliced.layersToClipperPaths()):
  #make the corners softer with a morphological opening (erosion+dilation)
  offset.do(aux1,  -openingRadius1, paths)
  offset.do(opened, openingRadius1, aux1)
  #skip carving a hole for the first layers
  if nlayer<bottomNumLayers:
    result = opened
  else:
    #erode + opening
    thickness = vaseThickness[nlayer-bottomNumLayers]
    offset.do(aux1, -thickness-openingRadius2, opened)
    offset.do(aux2,            openingRadius2, aux1)
    #substract the eroded slice to the original slice
    clipper.do(aux3, c.ctDifference, opened, aux2)
    #setLayerFromClipperObject() is faster if we pass it a ClipperPolyTree instead of a ClipperPaths
    result = aux3
  newsliced.setLayerFromClipperObject(nlayer, result)
  #shift the old object in X
  for path in paths:
    path[:,0] += shiftX
  sliced.setLayerFromClipperObject(nlayer, paths)

#show the original and the modified objects

if show3D:
  import pyslic3r.plot3d  as p3
  p3.mayaplotN([sliced, newsliced])

if show2D:
  import pyslic3r.plot2d  as p2
  p2.showSlices([sliced, newsliced], modeN=True, BB=[])

#write the objects to disk as sequences of ClipperPaths
#and Z values, ready to be read by the patched version
#of slic3r. The three Z values are required by slic3r's
#internal logic, and are the height, the upper boundary
#and the center of each slice, respectively.

#note that, instead of adding the modified slices to 'newsliced',
#we could have just made "aux2 = c.ClipperPaths()" instead of
#"aux2 = c.ClipperPolyTree()", and collect copies of aux2 (made
#with "aux2.copy()") in a list, so here we could just plug that
#list, instead of having to use again the generator
#newsliced.layersToClipperPaths()

pathsAndZss = [(paths, (layerHeight, z+layerHeight/2, z)) for paths,z in zip(newsliced.layersToClipperPaths(), zs)]

c.ClipperPathsAndZsToStream(len(pathsAndZss), pathsAndZss, 'vase.paths')

#now, if you have compiled the patched version of slic3r,
#just call it from the command line:

#perl path/to/slic3r/slic3r.pl --import-paths=path/to/data/vase.paths x

#this should generate the file /path/to/data/vase.gcode
