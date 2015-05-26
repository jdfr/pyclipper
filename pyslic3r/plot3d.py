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

import itertools as it

import numpy as n

import _SlicedModel as p

from mayavi import mlab

def mayaplot(slicedmodel, cmap='autumn', linecol=(0,0,0), showMesh=True, show=True):
  """use mayavi to plot a sliced model"""
  if isinstance(slicedmodel, p.SliceCollection):
    slicedmodel = slicedmodel.toSlicedModel()
  if not isinstance(slicedmodel, p.SlicedModel):
    raise Exception('only SlicedModel objects are supported')
  if showMesh:
    #plot surfaces
    ps, triangles = slicedmodel.layersAsTriangleMesh()
    mlab.triangular_mesh(ps[:,0], ps[:,1], ps[:,2], triangles, 
                         colormap=cmap, representation='surface')

  #make a list of pairs (cycle, z), composed from both contours and holes with their respective z's
  allcycles = list(it.chain.from_iterable( it.chain(((contour,z),), zip(holes, it.cycle((z,))))
                                           for _,_,z,contour,holes in slicedmodel.allExPolygons(asView=True)))
  #get cycle sizes    
  cyclessizes = list(cycle.shape[0] for cycle, z in allcycles)
  #get cumulative starting index for each cycle
  cyclestartidxs = n.roll(n.cumsum(cyclessizes), 1)
  cyclestartidxs[0] = 0
  #concatenate XY coords for all cycles
  #cyclesxy = n.vstack([cycle for cycle,_ in allcycles])
  cyclesx  = n.empty((sum(cyclessizes),))
  cyclesy  = n.empty((cyclesx.shape[0],))
  #size matrices for (a) concatenated z values and (b) line connections for all cycles
  cyclesz  = n.empty((cyclesx.shape[0],))
  conns    = n.empty((cyclesx.shape[0],2))
  #iterate over each cycle's starting index, size, and z
  for startidx, size, (cycle,z) in it.izip(cyclestartidxs, cyclessizes, allcycles):
    endidx = startidx+size
    cyclesx[startidx:endidx] = cycle[:,0]       #set x for the current cycle
    cyclesy[startidx:endidx] = cycle[:,1]       #set y for the current cycle
    cyclesz[startidx:endidx] = z                #set z for the current cycle
    rang = n.arange(startidx, endidx)
    conns[startidx:endidx,0] = rang    #set line connections for the current cycle
    conns[startidx, 1] = rang[-1]
    conns[startidx+1:endidx,1] = rang[:-1]
  #put all the processed data into mayavi
  cyclesx *= p.scalingFactor
  cyclesy *= p.scalingFactor
  src = mlab.pipeline.scalar_scatter(cyclesx,cyclesy,cyclesz)
  src.mlab_source.dataset.lines = conns # Connect them
  lines = mlab.pipeline.stripper(src) # The stripper filter cleans up connected lines
  mlab.pipeline.surface(lines, color=linecol)#, line_width=1)#, opacity=.4) # Finally, display the set of lines
  if show:
    mlab.show()

def mayaplotN(slicedmodels, title=None, showMesh=True, colormaps=None, linecolors=None):
  """use mayavi to plot the sliced model"""
  
  if not colormaps:
    colormaps = ['autumn', 'cool']
  if not linecolors:
    linecolors = [(0,0,0)]
  
  if title: mlab.figure(figure=title)
  for slicedmodel, cmap, linecol in it.izip(slicedmodels, it.cycle(colormaps), it.cycle(linecolors)):
    mayaplot(slicedmodel, cmap, linecol, showMesh=showMesh, show=False)
  mlab.show()
