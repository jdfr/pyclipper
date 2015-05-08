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
from warnings import warn

try:
  import numpy as n
except:
    raise ImportError('could not load NUMPY!')

try:
  import matplotlib.pyplot as plt
  from matplotlib.path import Path
  from matplotlib.patches import PathPatch
  import mpl_toolkits.mplot3d as m3
  import _SlicedModel as p
  import _Clipper as c
  
  def object2DToPatches(obj, sliceindex=0, facecolor='#cccccc', edgecolor='#999999'):
    """Universial conversion of objects to iterators of patches. It works for:
        -numpy.ndarray (bi-dimensional, if it is integer, it is scaled with pyslc3r.scalingFactor)
        -pyslic3r.ExPolygon
        -pyslic3r.Layer
        -pyslic3r.SlicedModel     (only the layer specified by sliceindex)
        -pyslic3r.SliceCollection (only the layer specified by sliceindex)
        -pyslic3r.ClipperPaths
        -pyslic3r.ClipperPolyTree
        -list of any of these objects
        """
    if isinstance(obj, n.ndarray):
      contour = obj
      return (PathPatch(contours2path([contour]), facecolor=facecolor, edgecolor=edgecolor),)
    if isinstance(obj, p.ExPolygon):
      expolygon = obj
      return (PathPatch(expolygon2path(expolygon.contour, expolygon.holes), facecolor=facecolor, edgecolor=edgecolor),)
    if isinstance(obj, p.Layer):
      layer = obj
      return (PathPatch(expolygon2path(exp.contour, exp.holes),
                        facecolor=facecolor, edgecolor=edgecolor)
              for exp in layer.expolygons)
    if isinstance(obj, p.SliceCollection):
      slicecollection = obj
      return object2DToPatches(slicecollection.slices[sliceindex], facecolor, edgecolor)
    if isinstance(obj, p.SlicedModel):
      slicedmodel = obj
      return (PathPatch(expolygon2path(exp.contour, exp.holes), facecolor=facecolor, edgecolor=edgecolor)
                 for exp in slicedmodel[sliceindex,:])
    elif isinstance(obj, c.ClipperPaths):
      clipperpaths = obj
      contours = list(x for x in clipperpaths)
      return (PathPatch(contours2path(contours), facecolor=facecolor, edgecolor=edgecolor),)
    elif isinstance(obj, c.ClipperPolyTree):
      return object2DToPatches(c.ClipperObjects2SlicedModel([obj], n.array(0.0)), facecolor, edgecolor)
    elif isinstance(obj, list) and all(isinstance(x, n.ndarray) for x in obj):
      contours = obj
      return (PathPatch(contours2path(contours), facecolor=facecolor, edgecolor=edgecolor),)
    elif hasattr(obj, '__next__'):
      return it.chain(object2DToPatches(x) for x in obj)
    else:
      raise Exception('Cannot convert this object type to patches: '+str(type(obj)))
      
  def show2DObject(obj, sliceindex=0, fig=None, facecolor='#cccccc', edgecolor='#999999'):
    """Universial show function for slice objects. It works for many kinds of objects,
       see objectToPatches() for a list"""
    minx = n.inf  
    miny = n.inf  
    maxx = -n.inf  
    maxy = -n.inf
    
    if fig is None:
      fig = plt.figure()
    ax = fig.add_subplot(111)
    for patch in object2DToPatches(obj, sliceindex, facecolor, edgecolor):
      ax.add_patch(patch)
      
      vs = patch.get_path().vertices
      
      vsmin = vs.min(axis=0)
      vsmax = vs.max(axis=0)
      
      minx = min(minx, vsmin[0])
      miny = min(miny, vsmin[1])
      maxx = max(maxx, vsmax[0])
      maxy = max(maxy, vsmax[1])
    cx = (maxx+minx)/2
    cy = (maxy+miny)/2
    dx = (maxx-minx)
    dy = (maxy-miny)
    
    maxd = max(dx, dy)*1.1
    
    ax.set_xbound(cx-maxd, cx+maxd)
    ax.set_ybound(cy-maxd, cy+maxd)
    plt.show()
  
  
  def contours2path(contours):
    sizes         = n.array([x.shape[0] for x in contours])
    accums        = n.cumsum(sizes[:-1])
    vertices      = n.vstack(contours)
    if vertices.dtype==n.int64:
      vertices    = vertices.astype(n.float64)*p.scalingFactor
    codes         = n.full((n.sum(sizes),), Path.LINETO, dtype=int)
    codes[0]      = Path.MOVETO
    codes[accums] = Path.MOVETO
    return Path(vertices, codes)
    
  def expolygon2path(contour, holes):
    """helper function for slices2Patches"""
  #  print 'MIRA: '
  #  print contour.max(axis=0)
  #  print contour.min(axis=0)
    allpols       = [contour]+holes
    return contours2path(allpols)
    
  
  def slices2Patches(slicedmodel, facecolor='#cccccc', edgecolor='#999999'):
    """helper function for showSlices3D"""
    paths     = ((z, expolygon2path(contour, holes)) for _, _, z, contour, holes in slicedmodel.allExPolygons())
    patches   = ((z, PathPatch(path, facecolor=facecolor, edgecolor=edgecolor)) for z, path in paths)
    return patches
    
  def showSlices3D(slicedmodel, fig=None, zfactor=1.0, facecolor='#cccccc', edgecolor='#999999'):
    """use matplotlib to render the slices. The rendering quality is exceptional;
    it is a shame that matplotlib has no proper 3d navigation support and no proper z buffer"""
    minx = n.inf  
    miny = n.inf  
    minz = n.inf  
    maxx = -n.inf  
    maxy = -n.inf
    maxz = -n.inf
    
    if fig is None:
      fig = plt.figure()
    ax = m3.Axes3D(fig)
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
except:
  warn('Could not load MATPLOTLIB. The functions that depend on it have not been defined')

try:
  from mayavi import mlab

  def mayaplot(slicedmodel, cmap='autumn', linecol=(0,0,0), showMesh=True, show=True):
    """use mayavi to plot a sliced model"""
    if isinstance(slicedmodel, p.SliceCollection):
      slicedmodel = slicedmodel.toSlicedModel()
    if not isinstance(slicedmodel, p.SlicedModel):
      raise Exception('only SlicedModel objects are supported')
    if showMesh:
      #plot surfaces
      ps, triangles = p.layersAsTriangleMesh(slicedmodel)
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
  
  def mayaplotN(slicedmodels, showMesh=True, colormaps=None, linecolors=None):
    """use mayavi to plot the sliced model"""
    
    if not colormaps:
      colormaps = ['autumn', 'cool']
    if not linecolors:
      linecolors = [(0,0,0)]
    
    for slicedmodel, cmap, linecol in it.izip(slicedmodels, it.cycle(colormaps), it.cycle(linecolors)):
      mayaplot(slicedmodel, cmap, linecol, showMesh=showMesh, show=False)
    mlab.show()
except:
  warn('Could not load MAYAVI. The functions that depend on it have not been defined')


