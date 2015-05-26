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

import _Clipper     as c

import matplotlib.pyplot      as plt
from   matplotlib.path    import Path
from   matplotlib.patches import PathPatch
from   matplotlib.lines   import Line2D

defaultPatchArgs  =   {'facecolor':'#cccccc', 'edgecolor':'#999999', 'lw':1}
defaultPatchArgss = [ {'facecolor':'#ff0000', 'edgecolor':'#000000', 'lw':1.5},
                      {'facecolor':'#0000ff', 'edgecolor':'#000000', 'lw':1},
                      {'facecolor':'#00ff00', 'edgecolor':'#000000', 'lw':0.75},
                      {'facecolor':'#00ffff', 'edgecolor':'#000000', 'lw':0.75},
                      {'facecolor':'#ffff00', 'edgecolor':'#000000', 'lw':0.75},
                      {'facecolor':'#ff00ff', 'edgecolor':'#000000', 'lw':0.75} ]

def object2DToPatches(obj, sliceindex=None, linestyle=None, patchArgs=defaultPatchArgs):
  """Universial conversion of objects to iterators of patches. It works for:
      -numpy.ndarray (bi-dimensional)
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
    contour         = obj
    paths           = (contours2path([contour]),)
  elif  hasattr(obj, 'contour') and hasattr(obj, 'holes'):    #isinstance(obj, _SlicedModel.ExPolygon):
    expolygon       = obj
    paths           = (expolygon2path(expolygon.contour, expolygon.holes),)
  elif  hasattr(obj, 'expolygons'): #isinstance(obj, _SlicedModel.Layer):
    layer           = obj
    paths           = (expolygon2path(exp.contour, exp.holes) for exp in layer.expolygons)
  elif  hasattr(obj, 'slices'):     #isinstance(obj, _SlicedModel.SliceCollection):
    slicecollection = obj
    if sliceindex is None: raise ValueError('if the first argument is a SliceCollection, second argument must be an index')
    return object2DToPatches(slicecollection.slices[sliceindex], linestyle=linestyle, patchArgs=patchArgs)
  elif  hasattr(obj, 'toExPolygonList'): #isinstance(obj, p.SlicedModel):
    if sliceindex is None: raise ValueError('if the first argument is a SlicedModel,     second argument must be an index')
    slicedmodel     = obj
    paths           = (expolygon2path(exp.contour, exp.holes) for exp in slicedmodel[sliceindex,:])
  elif isinstance(obj, c.ClipperPaths):
    clipperpaths    = obj
    contours        = (x for x in clipperpaths)
    paths           = (contours2path(contours),)
  elif isinstance(obj, c.ClipperPolyTree):
    return object2DToPatches(c.ClipperObjects2SlicedModel([obj], n.array(0.0)), linestyle=linestyle, patchArgs=patchArgs)
  elif hasattr(obj, '__getitem__'):
    if sliceindex is not None:
      return object2DToPatches(obj[sliceindex], linestyle=linestyle, patchArgs=patchArgs)
    if all(isinstance(x, n.ndarray) for x in obj):
      contours = obj
      paths    = (contours2path(contours),)
    else:
      raise Exception('if sliceindex is None and obj is indexable, all elements in obj must be arrays')
  elif hasattr(obj, '__next__'):
    return it.chain(object2DToPatches(x, linestyle=linestyle, patchArgs=patchArgs) for x in obj)
  else:
    raise Exception('Cannot convert this object type to patches: '+str(type(obj)))
    
  if linestyle is None:
    patches = ((PathPatch(path, **patchArgs), None) for path in paths)
  else:
    patches =  ((PathPatch(path, **patchArgs), Line2D(path.vertices[:,0], path.vertices[:,1], **linestyle)) for path in paths)
  
  return patches
  
    
def contours2path(contours, scalingFactor=0.000001):
  """helper function for object2DToPatches()"""
  contours      = [n.vstack((c, c[0,:])) for c in contours] #close the contours
  sizes         = n.array([x.shape[0] for x in contours])
  accums        = n.cumsum(sizes[:-1])
  vertices      = n.vstack(contours)
  if vertices.dtype==n.int64:
    vertices    = vertices*scalingFactor
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
  if hasattr(obj, 'getBoundingBox'): #isinstance(obj, p.SlicedModel):
    return obj.getBoundingBox()
  if   isinstance(obj, n.ndarray):
    minx, miny = obj.min(axis=0)
    maxx, maxy = obj.max(axis=0)
    return (minx, maxx, miny, maxy)
  if hasattr(obj, 'contour'):            return getBoundingBox(obj.contour)    #_SlicedModel.ExPolygon, no need to bother with the holes
  if hasattr(obj, 'expolygons'):         return getBoundingBox(obj.expolygons) #_SlicedModel.Layer
  if hasattr(obj, 'slices'):             return getBoundingBox(obj.slices)     #_SlicedModel.SliceCollection
  if isinstance(obj, c.ClipperPaths):    return getBoundingBox([x for x in obj])
  if isinstance(obj, c.ClipperPolyTree): return getBoundingBox(c.ClipperObjects2SlicedModel([obj], n.array(0.0)))
  if hasattr(obj, '__getitem__'):
    minxs, maxxs, minys, maxys = zip(*[getBoundingBox(x) for x in obj])
    minx = min(minxs)
    maxx = max(maxxs)
    miny = min(minys)
    maxy = max(maxys)
    return (minx, maxx, miny, maxy)
  raise Exception('Cannot compute the bounding box for object of type: '+str(type(obj)))

def show2DObject(obj, sliceindex=0, ax=None, linestyle=None, patchArgs=defaultPatchArgs, show=True, returnpatches=False):
  """Universal show function for slice objects. It works for many kinds of objects,
     see objectToPatches() for a list"""
  minx = n.inf  
  miny = n.inf  
  maxx = -n.inf  
  maxy = -n.inf
  
  if ax is None:
    fig     = plt.figure()
    ax      = fig.add_subplot(111, aspect='equal')
  patches   = object2DToPatches(obj, sliceindex, linestyle, patchArgs)
  if returnpatches:
    patches = list(patches)
  useline   = linestyle is not None
  for patch,line in patches:
    ax.add_patch(patch)
    if useline: ax.add_line(line)
    
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

def show2DObjectN(objs, sliceindexes=None, ax=None, linestyles=None, patchArgss=None, show=True, returnpatches=False):
  """use mayavi to plot a list of objects (mainly for SlicedModels, but should
  work for the others listed in objectToPatches()'s help"""
  
  if not sliceindexes:
    sliceindexes = [0]
  if not patchArgss:
    patchArgss = defaultPatchArgss
  if not linestyles:
    linestyles = [None]
  if ax is None:
    fig = plt.figure()
    ax  = fig.add_subplot(111, aspect='equal')
  allpatches = [None]*len(objs)
  
  for idx, (obj, sliceIndex, linestyle, patchArgs), in enumerate(it.izip(objs, it.cycle(sliceindexes), it.cycle(linestyles), it.cycle(patchArgss))):
    ps = show2DObject(obj, sliceindex=sliceIndex, ax=ax, linestyle=linestyle, patchArgs=patchArgs, show=False, returnpatches=returnpatches)
    if returnpatches:
      allpatches[idx] = ps
  if show:
    plt.show()
  if returnpatches:
    return allpatches


def showSlices(data, modeN=False, fig=None, ax=None, title=None, initindex=0, BB=-1, linestyle=None, patchArgs=None, show=True, handleEvents=True, scalingFactor=0.000001):
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
  #     -otherwise, we trust that it is a tuple (minx, maxx, miny, maxy), such as the tuple produced by getBoundingBox
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
    cx = (maxx+minx)/2.0*scalingFactor
    cy = (maxy+miny)/2.0*scalingFactor
    dx = (maxx-minx)    *scalingFactor
    dy = (maxy-miny)    *scalingFactor
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
    args    = lambda: dict(linestyles=linestyle, patchArgss=patchArgs, sliceindexes=index)
    def remove(allpatches):
      for patches in allpatches:
        for patch in patches:
          patch.remove()
          
  #setup fot single SlicedModel
  else:
    if patchArgs is None: patchArgs = defaultPatchArgs
    leng    = lambda x: len(x)
    showfun = show2DObject
    args    = lambda: dict(linestyle=linestyle,  patchArgs=patchArgs,  sliceindex=index[0])
    def remove(patches):
      for patch in patches:
        patch.remove()

  #final setup    
  index     = [min(max(0, initindex), leng(data)-1)] #this is a list as a workaround to set the value   index[0] in nested scopes

  #function to draw the figure      
  def paint():
    message = 'Layer %d/%d' % (index[0], leng(data)-1)
    #save the toolbar's view stack, which is reset when clearing axes or adding/removing objects
    if useBB and handleEvents: 
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
      if handleEvents:
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
  
  #finish setup  
  if handleEvents:  
    cid   = fig.canvas.mpl_connect('key_press_event', onpress)
  paint()
  if show:
    plt.show()
    
