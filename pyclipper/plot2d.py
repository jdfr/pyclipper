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

import itertools              as it

import numpy                  as n

from   . import Clipper       as c
from   . import minisix       as six

import matplotlib.pyplot      as plt
from   matplotlib.path    import Path
from   matplotlib.patches import PathPatch
from   matplotlib.lines   import Line2D
import matplotlib.collections as pc

def scaleObject(obj, scalingFactor=0.00000001):
  if isinstance(obj, n.ndarray):
    return obj*scalingFactor
  elif isinstance(obj, (c.ClipperPaths, list, tuple)):
    return [scaleObject(x, scalingFactor) for x in obj]
  else:
    raise Exception('Cannot recognize this object type: '+str(type(obj)))

def showOpenClipperPaths(paths, scalingFactor=0.00000001, fig=None, ax=None, show=True, title=None, **kwargs):
  if fig is None: fig = plt.figure(frameon=False)
  if ax  is None: ax  = fig.add_subplot(111, aspect='equal')
  """lines = scaleObject(paths, scalingFactor)
  lc = pc.LineCollection(lines, **kwargs)
  ax.add_collection(lc)
  ax.autoscale()
  if show:
    plt.show()"""
  for i, path in enumerate(paths):
    if len(path)==0:
      continue
    if path[0].dtype==n.int64:
      pathseq = [p*scalingFactor for p in path]
    else:
      pathseq = [p for p in path]
    pathc = pc.LineCollection(pathseq, colors=colorlist[i % len(colorlist)], **kwargs)
    ax.add_collection(pathc)
  ax.autoscale()
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  dd = max(abs(xlim[1]-xlim[0]), abs(ylim[1]-ylim[0]))
  shift = dd*0.1
  ax.set_xlim([xlim[0]-shift, xlim[1]+shift])
  ax.set_ylim([ylim[0]-shift, ylim[1]+shift])
  if title: fig.canvas.set_window_title(title)
  if show:
    plt.show()

colorlist = ['#cccccc', #gray
             '#ff0000', #red
             '#0000ff', #blue
             '#00ff00', #green
             '#00ffff', #cyan
             '#ffff00', #yellow
             '#ff00ff', #magenta
              ]

defaultPatchArgs  =   {'facecolor':colorlist[0], 'edgecolor':'#999999', 'lw':1}
defaultPatchArgss = [
                      {'facecolor':colorlist[1], 'edgecolor':'#000000', 'lw':1.5},
                      {'facecolor':colorlist[2], 'edgecolor':'#000000', 'lw':1},
                      {'facecolor':colorlist[3], 'edgecolor':'#000000', 'lw':0.75},
                      {'facecolor':colorlist[4], 'edgecolor':'#000000', 'lw':0.75},
                      {'facecolor':colorlist[5], 'edgecolor':'#000000', 'lw':0.75},
                      {'facecolor':colorlist[6], 'edgecolor':'#000000', 'lw':0.75},
                      ]

def object2DToPatches(obj, scalingFactor=0.00000001, sliceindex=None, linestyle=None, patchArgs=defaultPatchArgs):
  """Universial conversion of objects to iterators of patches. It works for:
      -numpy.ndarray (bi-dimensional)
      -ClipperPaths
      -ClipperDPaths
      -list/tuple of any of these objects except SlicedModel/SliceCollection (if sliceindex is not None, only the layer specified by sliceindex)
      -iterable   of any of these objects except SlicedModel/SliceCollection
    The result is always an iterable object with a sequence of patches
      """
  if obj==[] or obj==None:
      return []
  if isinstance(obj, n.ndarray):
    contour         = obj
    paths           = (contours2path([contour], scalingFactor=scalingFactor),)
  elif isinstance(obj, c.ClipperPaths):
    clipperpaths    = obj
    contours        = (x for x in clipperpaths)
    paths           = (contours2path(contours, scalingFactor=scalingFactor),)
  elif hasattr(obj, '__getitem__'):
    if sliceindex is not None:
      s = scalingFactor if not hasattr(obj, '__getitem__') else scalingFactor[sliceindex]
      return object2DToPatches(obj[sliceindex], scalingFactor=s, linestyle=linestyle, patchArgs=patchArgs)
    if all(isinstance(x, n.ndarray) for x in obj):
      contours = obj
      paths    = (contours2path(contours, scalingFactor=scalingFactor),)
    else:
      raise Exception('if sliceindex is None and obj is indexable, all elements in obj must be arrays')
  elif hasattr(obj, 'next'):
    ss = scalingFactor if hasattr(obj, '__getitem__') else it.cycle([scalingFactor])
    return it.chain(object2DToPatches(x, scalingFactor=s, linestyle=linestyle, patchArgs=patchArgs) for x, s in six.izip(obj, ss))
  else:
    raise Exception('Cannot convert this object type to patches: '+str(type(obj)))
    
  if linestyle is None:
    patches = ((PathPatch(path, **patchArgs), None) for path in paths)
  else:
    patches =  ((PathPatch(path, **patchArgs), Line2D(path.vertices[:,0], path.vertices[:,1], **linestyle)) for path in paths)
  
  return patches
  
def object2Lines(obj, scalingFactor=0.00000001, sliceindex=None, linestyle=None):
  if obj==[] or obj==None:
      return None
  if isinstance(obj, c.ClipperPaths):
    pathseq         = (x*scalingFactor for x in obj)
  elif isinstance(obj, c.ClipperDPaths):
    pathseq         = (x for x in obj)
  elif hasattr(obj, '__getitem__'):
    if sliceindex is not None:
      s = scalingFactor if not hasattr(obj, '__getitem__') else scalingFactor[sliceindex]
      return object2Lines(obj[sliceindex], linestyle=linestyle, scalingFactor=s)
    if all(isinstance(x, n.ndarray) for x in obj):
      pathseq       = (x for x in obj)
    else:
      raise Exception('if sliceindex is None and obj is indexable, all elements in obj must be arrays')
  elif hasattr(obj, 'next'):
    ss = scalingFactor if hasattr(obj, '__getitem__') else it.cycle([scalingFactor])
    return it.chain(object2Lines(x, linestyle=linestyle, scalingFactor=s) for x, s in six.izip(obj, ss))
  else:
    raise Exception('Cannot convert this object type to lines: '+str(type(obj)))
  if linestyle is None:
    return pc.LineCollection(pathseq)
  else:
    return pc.LineCollection(pathseq, **linestyle)
    
def contours2path(contours, scalingFactor=0.00000001):
  """helper function for object2DToPatches()"""
  contours        = [n.vstack((c, c[0,:])) for c in contours] #close the contours
  sizes           = n.array([x.shape[0] for x in contours])
  accums          = n.cumsum(sizes[:-1])
  if len(contours)==0:
    vertices      = n.empty((0,2))
  else:
    vertices      = n.vstack(contours)
  if vertices.dtype==n.int64:
    vertices      = vertices*scalingFactor
  codes           = n.full((n.sum(sizes),), Path.LINETO, dtype=int)
  if codes.size>0:
    codes[0]      = Path.MOVETO
    codes[accums] = Path.MOVETO
  return Path(vertices, codes)

def getBoundingBox(obj, scalingFactor=0.00000001):
  """get the bounding box for a variety of objects"""
  #if hasattr(obj, 'getBoundingBox'): #isinstance(obj, p.SlicedModel):
  #  return obj.getBoundingBox()
  if obj is None: return (n.inf, -n.inf, n.inf, -n.inf)
  if   isinstance(obj, n.ndarray):
    if len(obj)==0: return (n.inf, -n.inf, n.inf, -n.inf)
    minx, miny = obj.min(axis=0)
    maxx, maxy = obj.max(axis=0)
    return (minx, maxx, miny, maxy)
  if hasattr(obj, 'contour'):            return getBoundingBox(obj.contour)    #no need to bother with the holes
  if isinstance(obj, c.ClipperPaths):    return getBoundingBox([x*scalingFactor for x in obj])
  if isinstance(obj, c.ClipperDPaths):   return getBoundingBox([x for x in obj])
  if hasattr(obj, '__getitem__'):
    if len(obj)==0: return (n.inf, -n.inf, n.inf, -n.inf)
    minxs, maxxs, minys, maxys = zip(*[getBoundingBox(x) for x in obj])
    minx = min(minxs)
    maxx = max(maxxs)
    miny = min(minys)
    maxy = max(maxys)
    return (minx, maxx, miny, maxy)
  raise Exception('Cannot compute the bounding box for object of type: '+str(type(obj)))

def show2DObject(obj, scalingFactor=0.00000001, sliceindex=0, ax=None, usePatches=True, linestyle=None, patchArgs=defaultPatchArgs, show=True, returnpatches=False):
  """Universal show function for slice objects. It works for many kinds of objects,
     see objectToPatches() for a list"""
  """minx = n.inf  
  miny = n.inf  
  maxx = -n.inf  
  maxy = -n.inf"""
  
  if ax is None:
    fig     = plt.figure()
    ax      = fig.add_subplot(111, aspect='equal')
  if usePatches:
    objects   = object2DToPatches(obj, scalingFactor=scalingFactor, sliceindex=sliceindex, linestyle=linestyle, patchArgs=patchArgs)
    if returnpatches:
      objects = list(objects)
    useline   = linestyle is not None
    for patch,line in objects:
      ax.add_patch(patch)
      if useline: ax.add_line(line)
      
      """vs = patch.get_path().vertices
      
      if vs.size>0:
        vsmin = vs.min(axis=0)
        vsmax = vs.max(axis=0)
        
        minx = min(minx, vsmin[0])
        miny = min(miny, vsmin[1])
        maxx = max(maxx, vsmax[0])
        maxy = max(maxy, vsmax[1])"""
  else:
    objects = object2Lines(obj, scalingFactor=scalingFactor, sliceindex=sliceindex, linestyle=linestyle)
    if returnpatches:
      objects = [objects]
    if objects!=None:
      ax.add_collection(objects)
    """paths = linc.get_paths()
      for vs in paths:
        if vs.size>0:
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
  ax.set_ybound(cy-maxd, cy+maxd)"""
  if show:
    plt.show()
  if returnpatches:
    return objects

def show2DObjectN(objs, scalingFactor=None, sliceindexes=None, ax=None, usePatchess=None, linestyles=None, patchArgss=None, show=True, returnpatches=False):
  """use mayavi to plot a list of objects (mainly for SlicedModels, but should
  work for the others listed in objectToPatches()'s help"""

  if not scalingFactor:
    scalingFactor = it.cycle([0.00000001])
  if not sliceindexes:
    sliceindexes = [0]
  if not patchArgss:
    patchArgss = defaultPatchArgss
  if not linestyles:
    linestyles = [None]
  if not usePatchess:
    usePatchess = [True]
  if ax is None:
    fig = plt.figure()
    ax  = fig.add_subplot(111, aspect='equal')
  allpatches = [None]*len(objs)
  
  for idx, (obj, sliceIndex, linestyle, patchArgs, usePatches, ss), in enumerate(six.izip(objs, it.cycle(sliceindexes), it.cycle(linestyles), it.cycle(patchArgss), it.cycle(usePatchess), scalingFactor)):
    ps = show2DObject(obj, scalingFactor=ss, sliceindex=sliceIndex, ax=ax, linestyle=linestyle, patchArgs=patchArgs, show=False, usePatches=usePatches, returnpatches=returnpatches)
    if returnpatches:
      allpatches[idx] = ps
  if show:
    plt.show()
  if returnpatches:
    return allpatches


def showSlices(data, modeN=False, fig=None, ax=None, title=None, initindex=0, BB=-1, zs=None, usePatches=None, linestyle=None, patchArgs=None, show=True, handleEvents=True, scalingFactor=0.00000001):
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
  #     -if data is a list and BB is an empty list, we generate it from all the bounding boxes
  #     -otherwise, we trust that it is a tuple (minx, maxx, miny, maxy), such as the tuple produced by getBoundingBox
  if modeN:
    if type(scalingFactor)!=list:
      scalingFactor = [scalingFactor]*len(data)
  if type(BB)==int:
    if modeN:
      BB = getBoundingBox(data[BB], scalingFactor[BB])
    else:
      BB = getBoundingBox(data, scalingFactor)
  elif BB==[] and modeN:
    Bs = [getBoundingBox(d, s) for d, s in zip(data, scalingFactor)]
    BB = [fun([B[k] for B in Bs]) for k, fun in enumerate([min, max, min, max])]
  useBB      = BB is not None and not n.any(n.isinf(BB))

  #initial common setup
  if fig is None: fig = plt.figure(frameon=False)
  if ax  is None: ax  = fig.add_subplot(111, aspect='equal')
  if title: fig.canvas.set_window_title(title)
  patches    = [None]    #this is a list as a workaround to set the value patches[0] in nested scopes
  clearPatches = False  #hard-coded flag, change if you suspect ax.cla() is causing memory leaks
  txt   = ax.set_title('Layer X/X')
  if useBB:
    minx, maxx, miny, maxy = BB
    cx = (maxx+minx)/2.0
    cy = (maxy+miny)/2.0
    dx = (maxx-minx)
    dy = (maxy-miny)
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
    leng    = max([len(x) for x in data if not x is None])
    showfun = show2DObjectN
    args    = lambda: dict(scalingFactor=scalingFactor, usePatchess=usePatches, linestyles=linestyle, patchArgss=patchArgs, sliceindexes=index)
    def remove(allpatches):
      for patches in allpatches:
        for patch in patches:
          patch.remove()
          
  #setup fot single SlicedModel
  else:
    if patchArgs is None: patchArgs = defaultPatchArgs
    leng    = len(data)
    showfun = show2DObject
    args    = lambda: dict(scalingFactor=scalingFactor, usePatches=usePatches, linestyle=linestyle,  patchArgs=patchArgs,  sliceindex=index[0])
    def remove(patches):
      for patch in patches:
        patch.remove()

  #final setup    
  #index = [initindex]
  index     = [min(max(0, initindex), leng-1)] #this is a list as a workaround to set the value   index[0] in nested scopes

  #function to draw the figure      
  def paint():
    message = 'Layer %d/%d' % (index[0], leng-1)
    if zs is not None:
      message += ": %f" % zs[index[0]]
    #save the toolbar's view stack, which is reset when clearing axes or adding/removing objects
    if useBB and handleEvents: 
      t = fig.canvas.toolbar
      views = t._views
      poss  = t._positions
    #clear the figure and set the title text, depending on the method
    if clearPatches:
      if isinstance(patches[0], list):
        remove(patches[0])
      txt.set_text(message)
    else:
      #ax.cla()
      ax.clear()
      ax.set_title(message)
    #draw objects
    patches[0] = showfun(data, ax=ax, show=False, returnpatches=clearPatches, **args())
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
    l   = leng-1
    if   key == 'down'  and index[0]>0:
      index[0] -= 1
      paint()
    elif key == 'up'    and index[0]<l:
      index[0]  += 1
      paint()
    elif key == 'a':
      newindex = index[0] + int(l*0.1)
      if newindex<leng:
        index[0] = newindex
        paint()
    elif key == 'z':
      newindex = index[0] - int(l*0.1)
      if newindex>=0:
        index[0] = newindex
        paint()
  
  #finish setup  
  if handleEvents:  
    cid   = fig.canvas.mpl_connect('key_press_event', onpress)
    #by default, the most common operation is to zoom into some part of the plot
    fig.canvas.toolbar.zoom()
  paint()
  if show:
    plt.show()
    
