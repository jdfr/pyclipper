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




from mayavi import mlab

def mayaplot(slicedmodel):
  """use mayavi to plot the sliced model"""
  points, triangles = p.layersAsTriangleMesh(slicedmodel)
  
  mlab.triangular_mesh(points[:,0], points[:,1], points[:,2], triangles, representation='surface')
  mlab.show()

def mayaplot2(slicedmodel1, slicedmodel2, color1=(1,0,0), color2=(0,1,0)):
  """use mayavi to plot the sliced model"""
  points1, triangles1 = p.layersAsTriangleMesh(slicedmodel1)
  points2, triangles2 = p.layersAsTriangleMesh(slicedmodel2)
  
  rep = 'surface'
  
  mlab.triangular_mesh(points1[:,0], points1[:,1], points1[:,2], triangles1, color=color1, representation=rep)
  mlab.triangular_mesh(points2[:,0], points2[:,1], points2[:,2], triangles2, color=color2, representation=rep)
  mlab.show()



