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
import pyslic3r.plot3d  as p3

print """
A BRIEF INTRODUCTION TO pyslic3r.plot3d

pyslic3r.plot3d uses mayavi to render in 3D
objects of type SlicedModel.

PLEASE NOTE: in some cases mayavi may produce a
segmentation fault when the python interpreter
finishes!

There are two main functions:

mayaviplot(model, **kwargs) takes as input a SlicedModel,
and shows a 3D view of it. Line color and surface colormap
(each slice will be painted in a different color from
the colormap, according to its height as specified by
model.zs) can be specified as arguments.

mayaviplotN(models, **kwargs) is the same, but takes as
input a list of SliceModel objects.

In general, 3D plots give a more intuitive feeling of
the look and feel of the contours than 2D plots. It is,
however, more difficult to examine small details on the
slices. Also, matplotlib is more reliable than mayavi,
i.e., it is less likely to choke on malformed or
otherwise "peculiar" contours.
"""

mesh   = p.TriangleMesh('test.stl')
zs     = n.array([0.1, 5.1])
sliced = mesh.doslice(zs)

p3.mayaplot(sliced)

#Now, show two entities at the same time, the first
#one a SlicedMode, the second a list of ClipperPaths:

#displace in X a copy of slices
sliced2 = sliced.copy()
for k, paths in enumerate(sliced2.layersToClipperPaths()):
  for path in paths:
    path[:,0] += 50000000
  sliced2.setLayerFromClipperObject(k, paths)

#use the up/down arrow keys to navigate the slices
p3.mayaplotN([sliced, sliced2])
