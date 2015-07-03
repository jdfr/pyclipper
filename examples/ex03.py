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
import pyslic3r.plot2d  as p2

print """
A BRIEF INTRODUCTION TO pyslic3r.plot2d

pyslic3r.plot2d uses matplotlib to render in 2D
objects of various types (ClipperPaths, SlicedModel,
SliceCollection, Layer, Expolygon, sequences of
pairs of int64 values).

The main function is showSlices(), which can be used for
both online and offline rendering. In online rendering
of SlicedModel objects (and lists of ClipperPaths objects)
you can use the up/down arrow keys in order to navigate
the slices. It also preserves the axes's viewport
(as modified with matplotlib's pan & zoom options)
during slice navigation.

Slices are represented with matplotlib's lines and
patches, whose keyword arguments can be passed
as dictionaries to showSlices(), making for very
configurable render settings (colors, markers,
transparencies, etc.). See the source code for details.

With parameter modeN=False, showSlices() just shows one
object. However, if modeN=True, it expects its first
parameter to be a list of objects with many slices (for
example a SlicedModel or a list of ClipperPaths), which
are shown together. HOWEVER, these objects should have
identical numbers of layers, otherwise the behaviour is
undefined.
"""

mesh   = p.TriangleMesh('test.stl')
zs     = n.array([0.1, 5.1])
sliced = mesh.doslice(zs)

#use the up/down arrow keys to navigate the slices
p2.showSlices(sliced, modeN=False)

#Now, show two entities at the same time, the first
#one a SlicedMode, the second a list of ClipperPaths:

#displace in X a copy of slices
pathss = list(sliced.layersToClipperPaths())
for paths in pathss:
  for path in paths:
    path[:,0] += 100000000

#use the up/down arrow keys to navigate the slices
p2.showSlices([sliced, pathss], modeN=True, BB=[])
