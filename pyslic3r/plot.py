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

import _SlicedModel as p
import _Clipper     as c

try:
  import matplotlib.pyplot      as plt
  from   matplotlib.path    import Path
  from   matplotlib.patches import PathPatch

  defaultPatchArgs  =   {'facecolor':'#cccccc', 'edgecolor':'#999999', 'lw':1}
  defaultPatchArgss = [ {'facecolor':'#ff0000', 'edgecolor':'#000000', 'lw':1.5},
                        {'facecolor':'#0000ff', 'edgecolor':'#000000', 'lw':1},
                        {'facecolor':'#00ff00', 'edgecolor':'#000000', 'lw':0.75},
                        {'facecolor':'#00ffff', 'edgecolor':'#000000', 'lw':0.75},
                        {'facecolor':'#ffff00', 'edgecolor':'#000000', 'lw':0.75},
                        {'facecolor':'#ff00ff', 'edgecolor':'#000000', 'lw':0.75} ]
  
  def object2DToPatches(obj, sliceindex=None, patchArgs=defaultPatchArgs):
    """Universial conversion of objects to iterators of patches. It works for:
        -numpy.ndarray (bi-dimensional, if it is integer, it is scaled with pyslc3r.scalingFactor)
        -pyslic3r.ExPolygon
        -pyslic3r.Layer
        -pyslic3r.SlicedModel     (only the layer specified by sliceindex)
        -pyslic3r.SliceCollection (only the layer specified by sliceindex)
        -pyslic3r.ClipperPaths
        -pyslic3r.ClipperPolyTree
        -list/tuple of any of these objects except SlicedModel/SliceCollection (if sliceindex is not None, only the layer specified by sliceindex)
        -iterable   of any of these objects except SlicedModel/SliceCollection
      The result is always an iterable object with a sequence of patches
        """
    if isinstance(obj, n.ndarray):
      contour = obj
      return (PathPatch(contours2path([contour]), **patchArgs),)
    if isinstance(obj, p.ExPolygon):
      expolygon = obj
      return (PathPatch(expolygon2path(expolygon.contour, expolygon.holes), **patchArgs),)
    if isinstance(obj, p.Layer):
      layer = obj
      return (PathPatch(expolygon2path(exp.contour, exp.holes),
                        **patchArgs)
              for exp in layer.expolygons)
    if isinstance(obj, p.SliceCollection):
      slicecollection = obj
      if sliceindex is None: raise ValueError('if the first argument is a SliceCollection, second argument must be an index')
      return object2DToPatches(slicecollection.slices[sliceindex], patchArgs=patchArgs)
    if isinstance(obj, p.SlicedModel):
      slicedmodel = obj
      if sliceindex is None: raise ValueError('if the first argument is a SlicedModel,     second argument must be an index')
      return (PathPatch(expolygon2path(exp.contour, exp.holes), **patchArgs)
                 for exp in slicedmodel[sliceindex,:])
    elif isinstance(obj, c.ClipperPaths):
      clipperpaths = obj
      contours = list(x for x in clipperpaths)
      return (PathPatch(contours2path(contours), **patchArgs),)
    elif isinstance(obj, c.ClipperPolyTree):
      return object2DToPatches(c.ClipperObjects2SlicedModel([obj], n.array(0.0)), patchArgs=patchArgs)
    elif isinstance(obj, (list, tuple)):
      if sliceindex is not None:
        return object2DToPatches(obj[sliceindex], patchArgs=patchArgs)
      if all(isinstance(x, n.ndarray) for x in obj):
        contours = obj
        return (PathPatch(contours2path(contours), **patchArgs),)
      else:
        raise Exception('if sliceindex is None and obj is a list or tuple, all elements in obj must be arrays')
    elif hasattr(obj, '__next__'):
      return it.chain(object2DToPatches(x, patchArgs=patchArgs) for x in obj)
    else:
      raise Exception('Cannot convert this object type to patches: '+str(type(obj)))
      
  def contours2path(contours):
    """helper function for object2DToPatches()"""
    sizes         = n.array([x.shape[0] for x in contours])
    accums        = n.cumsum(sizes[:-1])
    vertices      = n.vstack(contours)
    if vertices.dtype==n.int64:
      vertices    = vertices*p.scalingFactor
    codes         = n.full((n.sum(sizes),), Path.LINETO, dtype=int)
    codes[0]      = Path.MOVETO
    codes[accums] = Path.MOVETO
    return Path(vertices, codes)

  def expolygon2path(contour, holes):
    """helper function for object2DToPatches"""
    allpols       = [contour]+holes
    return contours2path(allpols)

  def getBoundingBox(obj):
    """get the bounding box for a variety of objects"""
    if isinstance(obj, p.SlicedModel):
      return p.computeSlicedModelBBParams(obj)
    if   isinstance(obj, n.ndarray):
      minx, miny = obj.min(axis=0)
      maxx, maxy = obj.max(axis=0)
      return (minx, maxx, miny, maxy)
    if isinstance(obj, (list, tuple)):
      minxs, maxxs, minys, maxys = zip(*[getBoundingBox(x) for x in obj])
      minx = min(minxs)
      maxx = max(maxxs)
      miny = min(minys)
      maxy = max(maxys)
      return (minx, maxx, miny, maxy)
    if isinstance(obj, p.ExPolygon):
      return getBoundingBox(obj.contour) #no need to bother with the holes
    if isinstance(obj, p.Layer):
      return getBoundingBox(obj.expolygons)
    if isinstance(obj, p.SliceCollection):
      return getBoundingBox(obj.slices)
    if isinstance(obj, c.ClipperPaths):
      return  getBoundingBox([x for x in obj])
    if isinstance(obj, c.ClipperPolyTree):
      return getBoundingBox(c.ClipperObjects2SlicedModel([obj], n.array(0.0)))
    raise Exception('Cannot compute the bounding box for object of type: '+str(type(obj)))

  def show2DObject(obj, sliceindex=0, ax=None, patchArgs=defaultPatchArgs, show=True, returnpatches=False):
    """Universal show function for slice objects. It works for many kinds of objects,
       see objectToPatches() for a list"""
    minx = n.inf  
    miny = n.inf  
    maxx = -n.inf  
    maxy = -n.inf
    
    if ax is None:
      fig     = plt.figure()
      ax      = fig.add_subplot(111, aspect='equal')
    patches   = object2DToPatches(obj, sliceindex, patchArgs)
    if returnpatches:
      patches = list(patches)
    for patch in patches:
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
    if show:
      plt.show()
    if returnpatches:
      return patches
  
  def show2DObjectN(objs, sliceindexes=None, ax=None, patchArgss=None, show=True, returnpatches=False):
    """use mayavi to plot a list of objects (mainly for SlicedModels, but should
    work for the others listed in objectToPatches()'s help"""
    
    if not sliceindexes:
      sliceindexes = [0]
    if not patchArgss:
      patchArgss = defaultPatchArgss
    if ax is None:
      fig = plt.figure()
      ax  = fig.add_subplot(111, aspect='equal')
    allpatches = [None]*len(objs)
    
    for idx, (obj, sliceIndex, patchArgs), in enumerate(it.izip(objs, it.cycle(sliceindexes), it.cycle(patchArgss))):
      ps = show2DObject(obj, sliceindex=sliceIndex, ax=ax, patchArgs=patchArgs, show=False, returnpatches=returnpatches)
      if returnpatches:
        allpatches[idx] = ps
    if show:
      plt.show()
    if returnpatches:
      return allpatches
  

  def showSlices(data, modeN=False, fig=None, ax=None, title=None, initindex=0, BB=-1, patchArgs=None, show=True):
    """Advanced 2D viewer for objects representing sequences of slices (modeN=False)
    or lists/tuples of such objects to be paint at the same time (modeN=True). Use the
    up/down arrow keys to move up/down in the stack of slices. If provided
    with a bounding box (parameter BB=(cx, cy, dx, dy)), it maintains the axes
    limits to that bounding box, and it hacks the implementation of matplotlib's
    navigation toolbar to preserve the pan/zoom context across slice changes"""
    
    #detect if we are dealing with just a SlicedModel or a list of them
    if modeN and not isinstance(data, (list, tuple)):
      raise ValueError('If modeN==True, data should be a list or tuple')

    #setup of BB:
    #     -if it is None, we do not use it
    #     -if it is a number:
    #           *if data is a list, we set it to the bounding box of the corresponding elememnt in data
    #           *if data is a SlicedModel, we set it to its bounding box
    #     -otherwise, we trust that it is a tuple (minx, maxx, miny, maxy), such as the tuple produced by computeSlicedModelBBParams
    if type(BB)==int:
      if modeN:
        BB = getBoundingBox(data[BB])
      else:
        BB = getBoundingBox(data)
    useBB      = BB is not None

    #initial common setup
    if fig is None: fig = plt.figure(frameon=False)
    if ax  is None: ax  = fig.add_subplot(111, aspect='equal')
    if title: fig.canvas.set_window_title(title)
    patches    = [None]    #this is a list as a workaround to set the value patches[0] in nested scopes
    usePatches = False  #hard-coded flag, change if you suspect ax.cla() is causing memory leaks
    txt   = ax.set_title('Layer X/X')
    if useBB:
      minx, maxx, miny, maxy = BB
      cx = (maxx+minx)/2.0*p.scalingFactor
      cy = (maxy+miny)/2.0*p.scalingFactor
      dx = (maxx-minx)    *p.scalingFactor
      dy = (maxy-miny)    *p.scalingFactor
      fac            = 1.1
      minx = cx-dx*fac
      maxx = cx+dx*fac
      miny = cy-dy*fac
      maxy = cy+dy*fac
      ax.set_xlim(minx, maxx)
      ax.set_ylim(miny, maxy)

    #setup for list of SlicedModels
    if modeN:
      if patchArgs is None: patchArgs = defaultPatchArgss
      leng    = lambda x: len(x[0])
      showfun = show2DObjectN
      args    = lambda: dict(patchArgss=patchArgs, sliceindexes=index)
      def remove(allpatches):
        for patches in allpatches:
          for patch in patches:
            patch.remove()
            
    #setup fot single SlicedModel
    else:
      if patchArgs is None: patchArgs = defaultPatchArgs
      leng    = lambda x: len(x)
      showfun = show2DObject
      args    = lambda: dict(patchArgs=patchArgs,  sliceindex=index[0])
      def remove(patches):
        for patch in patches:
          patch.remove()

    #final setup    
    index     = [min(max(0, initindex), leng(data)-1)] #this is a list as a workaround to set the value   index[0] in nested scopes

    #function to draw the figure      
    def paint():
      message = 'Layer %d/%d' % (index[0], leng(data)-1)
      #save the toolbar's view stack, which is reset when clearing axes or adding/removing objects
      if useBB: 
        t = fig.canvas.toolbar
        views = t._views
        poss  = t._positions
      #clear the figure and set the title text, depending on the method
      if usePatches:
        if isinstance(patches[0], list):
          remove(patches[0])
        txt.set_text(message)
      else:
        #ax.cla()
        ax.clear()
        ax.set_title(message)
      #draw objects
      patches[0] = showfun(data, ax=ax, show=False, returnpatches=usePatches, **args())
      #set the toolbar view stack to the previous context
      if useBB: 
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        t = fig.canvas.toolbar
        t._views     = views
        t._positions = poss
        t._update_view()
      fig.canvas.draw()
      
    #function to receive keypress events to draw the figure      
    def onpress(event):
      key = str(event.key)
      l   = leng(data)-1
      if   key == 'down'  and index[0]>0:
        index[0] -= 1
        paint()
      elif key == 'up'    and index[0]<l:
        index[0]  += 1
        paint()
      elif key == 'left'  and index[0]>0:
        index[0]   = max(0, index[0]-5)
        paint()
      elif key == 'right' and index[0]<l:
        index[0]   = min(l, index[0]+5)
        paint()
    
    #finish setup    
    cid   = fig.canvas.mpl_connect('key_press_event', onpress)
    paint()
    if show:
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
except:
  warn('Could not load MAYAVI. The functions that depend on it have not been defined')


