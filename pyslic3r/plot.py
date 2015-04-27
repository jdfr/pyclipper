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

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import mpl_toolkits.mplot3d as m3
import _pyslic3r as p

def expolygon2path(contour, holes):
  """helper function for slices2Patches"""
  allpols       = [contour]+holes
#  print 'MIRA: '
#  print contour.max(axis=0)
#  print contour.min(axis=0)
  sizes         = n.array([x.shape[0] for x in allpols])
  accums        = n.cumsum(sizes[:-1])
  vertices      = n.vstack(allpols)
  codes         = n.full((n.sum(sizes),), Path.LINETO, dtype=int)
  codes[0]      = Path.MOVETO
  codes[accums] = Path.MOVETO
  return Path(vertices, codes)
  

def slices2Patches(slicedmodel, facecolor='#cccccc', edgecolor='#999999'):
  """helper function for showSlices3D"""
  paths     = ((z, expolygon2path(contour, holes)) for _, _, z, contour, holes in slicedmodel.allExPolygons())
  patches   = ((z, PathPatch(path, facecolor=facecolor, edgecolor=edgecolor)) for z, path in paths)
  return patches
  
def showSlices3D(slicedmodel, f=None, zfactor=1.0, facecolor='#cccccc', edgecolor='#999999'):
  """use matplotlib to render the slices. The rendering quality is exceptional;
  it is a shame that matplotlib has no proper 3d navigation support and no proper z buffer"""
  minx = n.inf  
  miny = n.inf  
  minz = n.inf  
  maxx = -n.inf  
  maxy = -n.inf
  maxz = -n.inf
  
  if f is None:
    f = plt.figure()
  ax = m3.Axes3D(f)
  for z, patch in slices2Patches(slicedmodel, facecolor, edgecolor):
    
    z *= zfactor
    ax.add_patch(patch)
    vs = patch.get_path().vertices
    m3.art3d.pathpatch_2d_to_3d(patch, z)
    
    vsmin = vs.min(axis=0)
    vsmax = vs.max(axis=0)
    
    minx = min(minx, vsmin[0])
    miny = min(miny, vsmin[1])
    minz = min(minz, z)
    maxx = max(maxx, vsmax[0])
    maxy = max(maxy, vsmax[1])
    maxz = max(maxz, z)
    
  cx = (maxx+minx)/2
  cy = (maxy+miny)/2
  cz = (maxz+minz)/2
  dx = (maxx-minx)
  dy = (maxy-miny)
  dz = (maxz-minz)
  
  maxd = max(dx, dy, dz)*1.1
  
  ax.set_xbound(cx-maxd, cx+maxd)
  ax.set_ybound(cy-maxd, cy+maxd)
  ax.set_zbound(cz-maxd, cz+maxd)
  
  plt.show()


def test():
  from mayavi import mlab
  import numpy as n
  x=[1,2,n.nan,3,4]; y=[1,2,n.nan,3,4]; z=[1,2,n.nan,3,4]
  mlab.plot3d(x,y,z)
  mlab.show()
  
def test():
  from mayavi import mlab
  import numpy as n
  x=[1,2,5,3,4]; y=[1,2,5,3,4]; z=[1,2,5,3,4]
  mlab.plot3d(x,y,z)
  mlab.show()
  

from mayavi import mlab

def mayaplot(slicedmodel, cmap='autumn', linecol=(0,0,0), show=True):
  """use mayavi to plot a sliced model"""
  #plot surfaces
  ps, triangles = p.layersAsTriangleMesh(slicedmodel)
  mlab.triangular_mesh(ps[:,0], ps[:,1], ps[:,2], triangles, 
                       colormap=cmap, representation='surface')

  #make a list of pairs (cycle, z), composed from both contours and holes with their respective z's
  allcycles = list(it.chain.from_iterable( it.chain(((contour,z),), zip(holes, it.cycle((z,))))
                                           for _,_,z,contour,holes in slicedmodel.allExPolygons()))
  #get cycle sizes    
  cyclessizes = list(cycle.shape[0] for cycle, z in allcycles)
  #get cumulative starting index for each cycle
  cyclestartidxs = n.roll(n.cumsum(cyclessizes), 1)
  cyclestartidxs[0] = 0
  #concatenate XY coords for all cycles
  cyclesxy = n.vstack([cycle for cycle,_ in allcycles])
  #size matrices for (a) concatenated z values and (b) line connections for all cycles
  cyclesz  = n.empty((cyclesxy.shape[0],))
  conns    = n.empty((cyclesxy.shape[0],2))
  #iterate over each cycle's starting index, size, and z
  for startidx, size, (_,z) in it.izip(cyclestartidxs, cyclessizes, allcycles):
    cyclesz[startidx:startidx+size] = z       #set z for the current cycle
    rang = n.arange(startidx, startidx+size)
    conns[startidx:startidx+size,0] = rang    #set line connections for the current cycle
    conns[startidx, 1] = rang[-1]
    conns[startidx+1:startidx+size,1] = rang[:-1]
  #put all the processed data into mayavi
  src = mlab.pipeline.scalar_scatter(cyclesxy[:,0],cyclesxy[:,1],cyclesz)
  src.mlab_source.dataset.lines = conns # Connect them
  lines = mlab.pipeline.stripper(src) # The stripper filter cleans up connected lines
  mlab.pipeline.surface(lines, color=linecol)#, line_width=1)#, opacity=.4) # Finally, display the set of lines
  if show:
    mlab.show()

def mayaplotN(slicedmodels, colormaps=None, linecolors=None):
  """use mayavi to plot the sliced model"""
  
  if not colormaps:
    colormaps = ['autumn', 'cool']
  if not linecolors:
    linecolors = [(0,0,0)]
  
  for slicedmodel, cmap, linecol in it.izip(slicedmodels, it.cycle(colormaps), it.cycle(linecolors)):
    mayaplot(slicedmodel, cmap, linecol, show=False)
  mlab.show()



