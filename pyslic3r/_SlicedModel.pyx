#cython: embedsignature=True


#while in development, we keep the libslic3rlib shared library in the very same
#directory as this file. In order to enable python to use this, execute this in
#the command line before entering python (it is necessary only once per bash session):
#    export LD_LIBRARY_PATH=.   <- or the relevant directory if we are not executing python from here
#more info: http://serverfault.com/questions/279068/cant-find-so-in-the-same-directory-as-the-executable
#
#ALTERNATIVE: A HACK IN __init__.py enables the use of the library without messing with the environment variables

cimport cython
from libcpp cimport bool

cimport numpy as cnp
import numpy as np

from slic3r_defs cimport *

from libc.stdio cimport *

cimport  Clipper as  c
cimport _Clipper as _c

from numpy.math cimport INFINITY#, NAN, isnan

import itertools as it

#np.import_array()
cnp.import_array()

cdef extern from "numpy/ndarraytypes.h" nogil:
  int NPY_ARRAY_CARRAY

cdef extern from "math.h" nogil:
  double fabs(double)

cimport cpython.ref as ref

cdef double INV_SCALING_FACTOR = 1.0/SCALING_FACTOR

scalingFactor    = SCALING_FACTOR
invScalingFactor = INV_SCALING_FACTOR

cdef object slice2xrange(int size, bool *justinteger, rang):
  """translate some python objects to a xrange"""
  cdef int start, stop, step
  justinteger[0] = False
  
  if rang is None:
    return xrange(size)
    
  elif isinstance(rang, slice):
    
    obj = rang.start
    if obj is None:
      start = 0
    elif isinstance(obj, int):
      start = obj
      start = max(0, start)
      start = min(start, size)
    else:
      raise IndexError('Invalid start index value')
      
    obj = rang.stop
    if obj is None:
      stop = size
    elif isinstance(obj, int):
      stop = obj
      stop = max(start, stop)      
      stop = min(stop, size)
    else:
      raise IndexError('Invalid stop index value')

    obj = rang.step
    if obj is None:
      step = 1
    else:
      step = obj
    return xrange(start, stop, step)
    
  elif isinstance(rang, int):
    start = rang
    if (start<0) or (start>=size):
      raise IndexError('Invalid index %d: %d' % (size, start))
    justinteger[0] = True
    return xrange(start, start+1)

  else:
    raise IndexError("Invalid indexing object")
  

#######################################################################
########## Sliced Model WRAPPER CLASS ##########
#######################################################################

def _rangecheck(int init, int end, int size):
  if init<0:
    raise IndexError('items for removal must be non-negative')
  if init>=size:
    raise IndexError('items for removal must be within the layer indexes')
  if end<init:
    raise IndexError('Invalid range')
  if end>=size:
    raise IndexError('last item for removal must be within the item indexes')

cdef class SlicedModelIterator:
  cdef SlicedModel model
  cdef size_t current
  def __cinit__(self, SlicedModel m):
    self.model   = m
    self.current = 0
  def __next__(self):
    if self.current >= self.model.thisptr[0].size():
      raise StopIteration
    else:
      x = self.model.toLayerList(True, True, self.current)
      self.current += 1
      return x

cdef class SlicedModel:
  """wrapper for the Slic3r data structure for a list of sliced layers"""
  
  ########################################################################
  #SPECIAL METHODS
  ########################################################################
  
  property zs:
    """expose the z values to python"""
    def __get__(self):
      return self.zvalues
    def __set__(self, cnp.ndarray[cnp.float64_t, ndim=1] val):
      self.zvalues = val

  def __cinit__(self, cnp.ndarray[cnp.float64_t, ndim=1] zvalues=None, bool doinit = True, bool doresize = False):
    if doinit:
      self.thisptr = new SLICEDMODEL()
    if doresize:
      self.thisptr[0].resize(zvalues.size)
    self.zvalues = zvalues

  def __dealloc__(self):
    del self.thisptr
  
  #quick and dirty (data is copied twice) pickle protocol
  #TODO: add support to load/save the data directly without intermediate structures like here
  def __reduce__(self):
    d = {'state': self.toSliceCollection(asView=True)}
    return (SlicedModel, (), d)
  def __setstate__(self, d):
    d['state'].toSlicedModel(self)
    
  def __len__(self):
    return self.thisptr[0].size()
  def __iter__(self):
    return SlicedModelIterator(self)
  
  def __getitem__(self, val):
    """rich but incomplete multidimension slicing support.
    
    This object represents a list of layers. Each layer has a z value and a
    list of expolygons. Each expolygon has a contour and a list of holes.
    Both contours and holes are lists of int64 points.
    
    Currently the following slicings are supported:
        * self[int]:             return Layer
        * self[slice]:           return [Layer]
        * self[int, 'z']:        return z value of the layer
        * self[int, int]:        return ExPolygon
        * self[int, slice]:      return [ExPolygon]
        * self[int, int, 'c']:   return array of coordinates of Contour
        * self[int, int, int]:   return array of coordinates of Hole
        * self[int, int, slice]: return list of Holes
      
      Slice limits are automatically clipped to be coherent with the object dimensions
      
      Note that it is not practical to slice the list/arrays here becuase of ambiguities
      (for example, self[int, int, int] may be either a Hole or a point in a Contour)
        """
    cdef size_t ndims, nlayer, nexp
    cdef bool asi = True
    cdef bool asv = True
    if isinstance(val, int) or isinstance(val, slice):
      #we use toLayerList instead of toSliceCollection to avoid an useless indirection layer
      return self.toLayerList(asi, asv, val)
    elif isinstance(val, tuple):
      ndims = len(val)
      if ndims==1:
        return self.toLayerList(asi, asv, val[0])
      else:
        if not isinstance(val[0], int):
          raise IndexError('multidimensional slicing across layers is not supported')
        else:
          nlayer = val[0]
          if ndims==2:
            if isinstance(val[1], basestring):
              return self.zvalues[nlayer]
            else:
              return self.toExPolygonList(nlayer, asi, asv, val[1])
          else:
            
            if not isinstance(val[1], int):
              raise IndexError('multidimensional slicing across ExPolygons is not supported')
            else:
              if ndims>3:
                raise IndexError('ExPolygon components are arrays. If you want to slice them, do it separately')
              nexp = val[1]
              if isinstance(val[2], basestring): #get the contour
                return self.Polygon2array(&self.thisptr[0][nlayer][nexp].contour, asi, asv)
              else:
                return self.toHoleList(nlayer, nexp, asi, asv, val[2])
    else:
      raise IndexError('Invalid slice object')

  ########################################################################
  #METHODS TO CONVERT TO/FROM Clipper datatypes
  ########################################################################

  @cython.boundscheck(False)
  cdef c.Paths* _layerToClipperPaths(self, size_t nlayer, c.Paths *output) nogil:
    if output==NULL:
      output = new c.Paths()
    Slic3rExPolygons_to_ClipperPaths(self.thisptr[0][nlayer], output)
    return output
  
  def layerToClipperPaths(self, size_t nlayer):
    """Transform a layer into a ClipperPaths structure, wich can be processed by Clipper"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    cdef _c.ClipperPaths paths = _c.ClipperPaths()
    paths.thisptr = self._layerToClipperPaths(nlayer, paths.thisptr)
    return paths

  def layersToClipperPaths(self):
    """Returns an iterator that transforms each layer into a ClipperPaths"""
    cdef _c.ClipperPaths paths 
    cdef size_t k
    for k in xrange(self.thisptr[0].size()):
      paths         = _c.ClipperPaths()
      paths.thisptr = self._layerToClipperPaths(k, paths.thisptr)
      yield paths
  
  @cython.boundscheck(False)
  cdef void _setLayerFromClipperPaths(self, size_t nlayer, c.Paths *inputs) nogil:
    """This operation is relatively expensive"""
    ClipperPaths_to_Slic3rExPolygons(inputs[0], &self.thisptr[0][nlayer], True)
  
  def setLayerFromClipperPaths(self, size_t nlayer, _c.ClipperPaths paths):
    """Transform a ClipperPaths structure back to a layer"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    self._setLayerFromClipperPaths(nlayer, paths.thisptr)
    
  @cython.boundscheck(False)
  cdef void _setLayerFromClipperPolyTree(self, size_t nlayer, c.PolyTree *inputs) nogil:
    """This operation is relatively cheap, compared to _setLayerFromClipperPaths()"""
    PolyTreeToExPolygons(inputs[0], &self.thisptr[0][nlayer], True)
  
  def setLayerFromClipperPolyTree(self, size_t nlayer, _c.ClipperPolyTree tree):
    """Transform a ClipperPolyTree structure back to a layer. Note that the method
    layerToClipperPolyTree does not exist, because Clipper provides no support for
    that conversion"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    self._setLayerFromClipperPolyTree(nlayer, tree.thisptr)

  def setLayerFromClipperObject(self, size_t nlayer, object obj):
    """Transform either a ClipperPaths or a ClipperPolyTree back to a layer.
    The former is more expensive than the latter"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if   isinstance(obj, _c.ClipperPaths   ): self._setLayerFromClipperPaths   (nlayer, (<_c.ClipperPaths>   obj).thisptr)
    elif isinstance(obj, _c.ClipperPolyTree): self._setLayerFromClipperPolyTree(nlayer, (<_c.ClipperPolyTree>obj).thisptr)
    else                                    : raise Exception('Cannot set layer from object of this type: '+str(type(obj)))
    
    
  ########################################################################
  #METHODS TO CONVERT TO PYTHONIC STRUCTURES
  ########################################################################

  @cython.boundscheck(False)
  def toSliceCollection(self, bool asInteger=False, bool asView=False, rang=None):
    """Same as toLayerList, but returns the list of layers wrapped in a SliceCollection
    object. This will be useful if SliceCollection acquires more attributes,
    for example if SCALING_FACTOR is un-hardcoded and moved to SlicedModel"""
    if isinstance(rang, int): #make sure that toLayerList() returns a list instead of a naked Layer
      rang = slice(rang, rang+1, 1)
    return SliceCollection(self.toLayerList(asInteger, asView, rang))
      
  @cython.boundscheck(False)
  cpdef object toLayerList(self, bool asInteger=False, bool asView=False, rang=None):
    """return a full-fledged pythonic representation of the layers of this
    SlicedModel object, as a list of Layer objects (all attributes fully pythonic,
    so it is easier to manipulate in python (removing and adding things
    arbitrarily).
    
    Each Layer object has a z value and a list of ExPolygon objects.
    
    An ExPolygon object represents a contour with holes. No nesting is allowed,
    i.e., there are no contours within the holes (however, nesting is implemented
    in practice as a list of logically unrelated but geometrically nested ExPolygons).
    
    Each ExPolygon has a contour and a list of holes.
    
    Contours and holes are numpy arrays of 2d points, each one representing a
    polygon."""
    
    cdef bool justInteger
    rango = slice2xrange(self.thisptr[0].size(), &justInteger, rang)
    
    ret =  [Layer(self.zvalues[nlayer],
                  self._toExPolygonList(nlayer, asInteger, asView, None))
            for nlayer in rango]
    if justInteger:
      return ret[0]
    return ret

  @cython.boundscheck(False)  
  cpdef object toExPolygonList(self, size_t nlayer, bool asInteger=False, bool asView=False, rang=None):
    """same as toLayerList(), but for ExPolygons"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    return self._toExPolygonList(nlayer, asInteger, asView, rang)

  @cython.boundscheck(False)  
  cdef object _toExPolygonList(self, size_t nlayer, bool asInteger=False, bool asView=False, rang=None):
    """same as toLayerList(), but for ExPolygons"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    cdef bool justInteger
    rango = slice2xrange(self.thisptr[0][nlayer].size(), &justInteger, rang)
    ret =  [ExPolygon(self.Polygon2array(&self.thisptr[0][nlayer][nexp].contour,      asInteger, asView),
                      self._toHoleList(nlayer, nexp, asInteger, asView, None))
                   for nexp in rango]
    if justInteger:
      return ret[0]
    return ret

  @cython.boundscheck(False)  
  cpdef object toHoleList(self, size_t nlayer, size_t nexp, bool asInteger=False, bool asView=False, rang=None):
    """same as toLayerList(), but for holes"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if   nexp>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect Expolygon ID')
    return self._toHoleList(nlayer, nexp, asInteger, asView, rang)
    
  @cython.boundscheck(False)  
  cdef object _toHoleList(self, size_t nlayer, size_t nexp, bool asInteger=False, bool asView=False, rang=None):
    """same as toLayerList(), but for holes"""
    cdef bool justInteger
    rango = slice2xrange(self.thisptr[0][nlayer][nexp].holes.size(), &justInteger, rang)
    ret =  [self.Polygon2array(&self.thisptr[0][nlayer][nexp].holes[nhole], asInteger, asView)
               for nhole in rango]
    if justInteger:
      return ret[0]
    return ret

  ########################################################################
  #METHODS TO ACCESS DATA
  ########################################################################

  @cython.boundscheck(False)
  cdef cnp.ndarray Polygon2array(self, Polygon *pol, bool asInteger=True, bool asView=True):
    """common code for getting arrays from contours and holes"""
    if asView:
      return Polygon2arrayView(self, pol)
    else:
      if asInteger:
        return Polygon2arrayI(pol)
      else:
        return Polygon2arrayF(pol)

  @cython.boundscheck(False)  
  def contour(self, size_t nlayer, size_t nExpolygon, bool asInteger=False, bool asView=False):
    """contour (as an array representing a list of points) of an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect Expolygon ID')
    return self.Polygon2array(&self.thisptr[0][nlayer][nExpolygon].contour, asInteger, asView)
    
  @cython.boundscheck(False)  
  def hole(self, size_t nlayer, size_t nExpolygon, size_t nhole, bool asInteger=False, bool asView=False):
    """hole (as an array representing a list of points) of an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect Expolygon ID')
    if nhole>=self.thisptr[0][nlayer][nExpolygon].holes.size():
      raise IndexError('incorrect hole ID')
    return self.Polygon2array(&self.thisptr[0][nlayer][nExpolygon].holes[nhole], asInteger, asView)

  @cython.boundscheck(False)
  def allExPolygons(self, bool asInteger=False, asView=False):
    """return a generator for all expolygons in all layers. Each ExPolygon is
    returned as a tuple with a layer index, an expolygon index (within the layer),
    a layer depth (z value), a contour and a list of holes. The contour and the
    holes are returned as numpy arrays, whose type is either numpy.int64 (native
    type of ExPolygon coordinates) or scaled numpy.float64 values"""
    cdef size_t k1, k2
    cdef double z
    cdef cnp.ndarray contour
    for k1 in xrange(self.thisptr[0].size()):
      z = self.zvalues[k1] #using yield, we cannot use a numpy array buffer declaration
      for k2 in xrange(self.thisptr[0][k1].size()):
        contour =  self.Polygon2array(&self.thisptr[0][k1][k2].contour,  asInteger, asView)
        holes   = [self.Polygon2array(&self.thisptr[0][k1][k2].holes[h], asInteger, asView)
                       for h in xrange(self.thisptr[0][k1][k2].holes.size())]
        yield (k1, k2, z, contour, holes)

  ########################################################################
  #METHODS TO REMOVE DATA
  ########################################################################

  #nogil SHOULD BE ALLOWED IN THE DEFINITIONS OF THE _removeXXX methods, BUT
  #WE CANNOT PUT IT BECAUSE OF A WEIRD CYTHON BUG (COMPILATION FAILS, COMPLAINING  
  #IN THE ARGUMENT LIST OF .remove(): "Converting to Python object not allowed without gil"
  
  @cython.boundscheck(False)  
  cdef void _removeLayers(self, size_t init, size_t end) nogil:
    cdef vector[ExPolygons].iterator it = self.thisptr[0].begin()
    self.thisptr[0].erase(it+init, it+end)
    
  @cython.boundscheck(False)  
  def removeLayers(self, size_t init, size_t end):
    """remove a range of layers"""
    _rangecheck(init, end, self.thisptr[0].size())
    self._remove(init, end)
    
  @cython.boundscheck(False)  
  cdef void _removeExPolygons(self, size_t nlayer, size_t init, size_t end) nogil:
    cdef vector[_ExPolygon].iterator it = self.thisptr[0][nlayer].begin()
    self.thisptr[0][nlayer].erase(it+init, it+end)
    
  @cython.boundscheck(False)  
  def removeExPolygons(self, size_t nlayer, size_t init, size_t end):
    """in a layer, remove a range of ExPolygons"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    _rangecheck(init, end, self.thisptr[0][nlayer].size())
    self._removeExPolygons(nlayer, init, end)
    
  @cython.boundscheck(False)  
  cdef void _removeHoles(self, size_t nlayer, size_t nexp, size_t init, size_t end) nogil:
    cdef vector[Polygon].iterator it = self.thisptr[0][nlayer][nexp].holes.begin()
    self.thisptr[0][nlayer][nexp].holes.erase(it+init, it+end)
    
  @cython.boundscheck(False)  
  def removeHoles(self, size_t nlayer, size_t nexp, size_t init, size_t end):
    """in an expolygon within a layer, remove a range of holes"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if nexp>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect ExPolygon ID')
    _rangecheck(init, end, self.thisptr[0][nlayer][nexp].holes.size())
    self._removeExPolygons(nlayer, init, end)

  ########################################################################
  #MISCELLANEA OF METHODS
  ########################################################################

  def __copy__(self):
    """return a copy"""
    cdef SlicedModel out = SlicedModel(self.zvalues)
    out.thisptr[0] = self.thisptr[0]
    return out
    
  def copyEmpty(self):
    """return a SliceModel with the same (but empty) layers then self"""
    return SlicedModel(self.zvalues, doresize=True)

  @cython.boundscheck(False)  
  def select(self, cnp.ndarray[cnp.int64_t, ndim=1] selectedzs):
    """given an array of layer indexes, returns a new SlicedModel with a copy of
    those layers"""
    cdef SlicedModel selected = SlicedModel(self.zvalues[selectedzs])
    cdef bool ok = True
    cdef size_t k, siz
    siz = selectedzs.size
    with nogil:
      selected.thisptr[0].reserve(siz)
      for k in range(siz):
        if (selectedzs[k]>=0) and ((<size_t>selectedzs[k])<self.thisptr[0].size()):
          selected.thisptr[0].push_back(self.thisptr[0][selectedzs[k]])
        else:
          ok = False
          break
    if not ok:
      raise Exception('Invalid layer index')
    return selected

  def merge(self, SlicedModel other, double mergeTolerance = 0.0):
    """merge data from this SlicedModel and another one into a new one.
    Slices from each model are merged if their z values are within mergeTolerance.
    WARNING: no sanity checks are done. If the ExPolygons within mergeTolerance
    interesect, the behaviour is undefined for further calls to the Slic3r C++ library."""
    return mergeSlicedModels([self, other], mergeTolerance)

  def save(self, basestring filename, basestring mode='ply'):
    model = mode.lower()
    if mode=='ply':
      writeAsPLY(self, filename)
    elif mode=='svg':
      writeAsSVG(self, filename)
    else:
      raise Exception('mode not understood: '+mode)
      
  cdef bool slicesAreOrdered(self):
    return (np.diff(self.zvalues)>=0).all()
    
  @cython.boundscheck(False)  
  cpdef size_t numLayers(self):
    """number of layers of the sliced model"""
    return self.thisptr[0].size()
  
  @cython.boundscheck(False)  
  cpdef size_t numExPolygons(self, size_t nlayer):
    """number of ExPolygons in a layer of the sliced model"""
    if nlayer>=self.thisptr.size():
      raise IndexError('incorrect layer ID')
    return self.thisptr[0][nlayer].size()
    
  @cython.boundscheck(False)  
  cpdef size_t numHoles(self, size_t nlayer, size_t nExpolygon):
    """number of holes in an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect Expolygon ID')
    return self.thisptr[0][nlayer][nExpolygon].holes.size()

#######################################################################
########## MERGING SEVERAL SlicedModels TOGETHER ##########
#######################################################################


@cython.boundscheck(False)  
def mergeSlicedModels(inputs, double mergeTolerance = 0.0):
  """merge data from a list of SlicedModels into a new one.
  Slices from each model are merged if their z values are within mergeTolerance.
  
  WARNING: may produce unexpected results if the distances between slices in
  any of the SlicedModels is lowerthan the tolerance
  
  WARNING: no boolean operation is performed. If the slices intersect,
  unexpected errors will likely follow"""
  
  #convert the input to a list
  cdef size_t inew, nmodels, reservesize, k, idxlowest, numToMerge
  cdef cnp.npy_intp allsizes
  cdef bool goon
  cdef double lowest, val
  cdef cnp.ndarray[cnp.float64_t, ndim=1] newzs
  cdef SLICEDMODEL *newptr = new SLICEDMODEL()
  cdef vector[int] numlayerss, idxs
  cdef vector[bool] ended, toMerge
  cdef vector[cnp.float64_t] currentzs
  cdef vector[cnp.float64_t*] zvaluess
  cdef vector[SLICEDMODEL*] thisptrs
  cdef SlicedModel model
  
  inputs = list(inputs)  
  allsizes = 0
  nmodels  = len(inputs)
  numlayerss.resize(nmodels)
  ended.resize(nmodels)
  toMerge.resize(nmodels)
  idxs.resize(nmodels)
  currentzs.resize(nmodels)
  zvaluess.resize(nmodels)
  thisptrs.resize(nmodels)
  
  #initialization, sanity checks
  for k in range(nmodels):
    model         = inputs[k]
    if not model.slicesAreOrdered():
      raise Exception('model %d is not ordered!' % k)
    zvaluess[k]   = <cnp.float64_t*>model.zvalues.data
    thisptrs[k]   = model.thisptr
    numlayerss[k] = model.thisptr[0].size()
    allsizes     += numlayerss[k]
    idxs[k]       = 0
    ended[k]      = numlayerss[k]==0

  newptr[0].reserve(allsizes)
  newzs     = cnp.PyArray_EMPTY(1, &allsizes, cnp.NPY_FLOAT64, 0)
  inew      = 0
  #make sure that we start the main loop only if we are going to do some work
  goon      = allsizes>0
  
  with nogil:
    while goon:
      
      #FIND LOWEST CURRENT SLICE ACROSS MODELS
      lowest    = INFINITY
      idxlowest = -1
      for k in range(nmodels):
        if not ended[k]:
          currentzs[k] = zvaluess[k][idxs[k]]
          if currentzs[k] < lowest:
            idxlowest = k
            lowest = currentzs[k]
      #assert idxlowest>=0
      
      #FIND SLICES WITHIN THE TOLERANCE OF THE CURRENT LOWEST, PREPARE TO MERGE THEM LATER
      val = numToMerge = reservesize = 0
      for k in range(nmodels):
        #for some cythonic reason, I cannot assign the boolean expression directly to toMerge[k]
        if (not ended[k]) and (fabs(currentzs[idxlowest]-currentzs[k])<=mergeTolerance):
          toMerge[k] = True
          numToMerge  += 1
          val         += currentzs[k]
          reservesize += thisptrs[k][0][idxs[k]].size()
        else:
          toMerge[k] = False
      
      if numToMerge==1: #no merging: just copy the lowest slice
      
        newptr[0].push_back(thisptrs[idxlowest][0][idxs[idxlowest]])
        newzs[inew]         = lowest
        inew               += 1
        idxs[idxlowest]  += 1
        ended[idxlowest]  = idxs[idxlowest]>=numlayerss[idxlowest]
      
      else: #merge the slices within tolerance
        
        newzs[inew] = val / numToMerge
        newptr[0].resize(inew+1)
        newptr[0][inew].reserve(reservesize)
        for k in range(nmodels):
          if toMerge[k]:
            newptr[0][inew].insert(newptr[0][inew].end(),  thisptrs[k][0][idxs[k]].begin(),  thisptrs[k][0][idxs[k]].end())
            idxs[k] += 1
            ended[k] = idxs[k]>=numlayerss[k]
        inew   += 1
  
      #termination condition: no more slices to add
      goon      = False
      for k in range(nmodels):
        if not ended[k]:
          goon = True
          break
        
  #if any merge happened, remove empty space at the end of zvalues
  if inew<newzs.size:
    newzs = newzs[:inew]
  
  #create new SlicedModel
  model = SlicedModel(newzs, False) #use doinit==False to avoid allocating an empty thisptr in the new object
  model.thisptr = newptr
  return model

#######################################################################
########## TRIANGULATION OF SlicedModel ##########
#######################################################################

@cython.boundscheck(False)
cdef void countPolygons(vector[vector[Polygons]] * polss, size_t *rnumP, size_t *rnumV) nogil:
  """compute the number of vertices and polygons in a vector of vectors of vectors of type Polygon"""
  cdef size_t numV = 0
  cdef size_t numP = 0
  cdef size_t k1, k2, k3
  for k1 in range(polss[0].size()):
    for k2 in range(polss[0][k1].size()):
      for k3 in range(polss[0][k1][k2].size()):
        numP += 1
        numV += polss[0][k1][k2][k3].points.size()
  rnumP[0] = numP  
  rnumV[0] = numV


@cython.boundscheck(False)  
cdef vector[vector[Polygons]] * triangulateAllLayers(SlicedModel model) nogil:
  """generate a model of the layers apt to be represented in a 3D view"""
  cdef size_t k1, k2, nlayers, nexpols
  cdef vector[vector[Polygons]] * polss = new vector[vector[Polygons]]()
  nlayers = model.thisptr[0].size()
  polss[0].resize(nlayers)
  for k1 in range(nlayers):
    nexpols = model.thisptr[0][k1].size()
    polss[0][k1].resize(nexpols)
    for k2 in range(nexpols):
      model.thisptr[0][k1][k2].triangulate_pp(&polss[0][k1][k2])
  return polss

@cython.boundscheck(False)
def layersAsTriangleMesh(SlicedModel model):
  """return an array of points and an array of triangles, STL style (i. e.,
  the points are not reused for neighbouring triangles)"""
  cdef vector[vector[Polygons]] * polss
  cdef Points * polpoints
  cdef cnp.ndarray[cnp.float64_t, ndim=2] points
  cdef cnp.ndarray[cnp.int64_t, ndim=2] triangles
  cdef size_t numP, numV, k1, k2, k3, k4, kp#, kt
  cdef cnp.ndarray[cnp.float64_t, ndim=1] zvalues = model.zvalues
  cdef bool ok = True
  kp = 0
  #kt = 0

  polss = triangulateAllLayers(model)
  try:
    countPolygons(polss, &numP, &numV)
    points    = np.empty((numV, 3), dtype=np.float64)
    #triangles = np.empty((numP, 3), dtype=np.int64)
    triangles = np.arange(numV).reshape((-1, 3))
    for k1 in range(polss[0].size()):
      z = zvalues[k1]
      for k2 in range(polss[0][k1].size()):
        for k3 in range(polss[0][k1][k2].size()):
          polpoints = &polss[0][k1][k2][k3].points
          if polpoints[0].size()!=3:
            raise Exception("Invalid triangulation!")
          for k4 in range(3):
            points[kp, 0]    = polpoints[0][k4].x*SCALING_FACTOR
            points[kp, 1]    = polpoints[0][k4].y*SCALING_FACTOR
            points[kp, 2]    = z
            #triangles[kt,k4] = kp
            kp += 1
          #kt += 1
    return (points, triangles)
  finally:
    del polss

#######################################################################
########## WRITING TO DISK SlicedModel ##########
#######################################################################
  
@cython.boundscheck(False)  
cdef void writeAsSVG(SlicedModel model, basestring filename):
  """write a SVG file in the style of slic3r --export-svg"""
  cdef size_t k1, k2, k3
  cdef double z, cx, cy, dx, dy, sx, sy
  cdef char space
  cdef FILE *f = fopen(filename, "w")
  cdef cnp.ndarray[cnp.float64_t, ndim=1] zvalues = model.zvalues
  cx, cy, dx, dy = computeSlicedModelBBParams(model)
  sx = cx-dx/2
  sy = cy-dy/2
  if f==NULL:
    raise IOError('Could not open the file in write mode')
  try:
    with nogil:
      #header
      fprintf(f, """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
<svg width="%f" height="%f" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:slic3r="http://slic3r.org/namespaces/slic3r">
<!-- 
Generated using pyslic3r pre-alpha
 -->
""", dx, dy)
      #layers
      for k1 in range(model.thisptr[0].size()):
        z = zvalues[k1]
        #layer prefix
        fprintf(f, '  <g id="layer%d" slic3r:z="%f">\n', k1, z)
        #expolygons
        for k2 in range(model.thisptr[0][k1].size()):
          #contour
          writePolygonSVG(&model.thisptr[0][k1][k2].contour, f, True, sx, sy)
          #holes
          for k3 in range(model.thisptr[0][k1][k2].holes.size()):
            writePolygonSVG(&model.thisptr[0][k1][k2].holes[k3], f, False, sx, sy)
        #layer postfix
        fputs('  </g>\n', f)
      #close svg
      fputs('</svg>', f)
  finally:
    fclose(f)
            
            
@cython.boundscheck(False)  
cdef void writeAsPLY(SlicedModel model, basestring filename):
  """write a sliced model to a PLY file"""
  cdef vector[vector[Polygons]] * polss
  cdef size_t numV, numP, numpoints, basecount, k1, k2, k3, k4
  cdef double z
  cdef FILE *f
  cdef size_t count = 0
  cdef cnp.ndarray[cnp.float64_t, ndim=1] zvalues = model.zvalues
  polss = triangulateAllLayers(model)
  try:
    f = fopen(filename, "w")
    if f==NULL:
      raise IOError('Could not open the file in write mode')
    try:
      with nogil:
        countPolygons(polss, &numP, &numV)
        #header
        fprintf(f, 'ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nelement face %d\nproperty list uchar int vertex_index\nend_header\n', numV, numP)
        #points
        for k1 in range(polss[0].size()):
          z = zvalues[k1]
          for k2 in range(polss[0][k1].size()):
            for k3 in range(polss[0][k1][k2].size()):
              for k4 in range(polss[0][k1][k2][k3].points.size()):
                fprintf(f, '%f %f %f\n', polss[0][k1][k2][k3].points[k4].x*SCALING_FACTOR, polss[0][k1][k2][k3].points[k4].y*SCALING_FACTOR, z)
        #polygons
        for k1 in range(polss[0].size()):
          for k2 in range(polss[0][k1].size()):
            for k3 in range(polss[0][k1][k2].size()):
              numpoints = polss[0][k1][k2][k3].points.size()
              fprintf(f, '%d', numpoints)
              for k4 in range(numpoints):
                fprintf(f, ' %d', count+k4)
              fputs('\n', f)
              count += numpoints
    finally:
      fclose(f)
  finally:
    del polss


@cython.boundscheck(False)
def computeSlicedModelBBParams(SlicedModel model):  
  """Compute some parameters of the bounding box: the center and the size"""
  cdef size_t k1, k2, k3, k4
  cdef double minx, maxx, miny, maxy, x, y, cx, cy, dx, dy
  
  minx = miny =  INFINITY
  maxx = maxy = -INFINITY
  for k1 in range(model.thisptr[0].size()):
    for k2 in range(model.thisptr[0][k1].size()):
      for k3 in range(model.thisptr[0][k1][k2].contour.points.size()):
        x = model.thisptr[0][k1][k2].contour.points[k3].x
        y = model.thisptr[0][k1][k2].contour.points[k3].y
        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)
      for k3 in range(model.thisptr[0][k1][k2].holes.size()):
        for k4 in range(model.thisptr[0][k1][k2].holes[k3].points.size()):
          x = model.thisptr[0][k1][k2].holes[k3].points[k4].x
          y = model.thisptr[0][k1][k2].holes[k3].points[k4].y
          minx = min(minx, x)
          miny = min(miny, y)
          maxx = max(maxx, x)
          maxy = max(maxy, y)
  cx = (maxx+minx)/2*SCALING_FACTOR
  cy = (maxy+miny)/2*SCALING_FACTOR
  dx = (maxx-minx)*SCALING_FACTOR
  dy = (maxy-miny)*SCALING_FACTOR
  #return (minx, maxx, miny, maxy)
  return (cx, cy, dx, dy)

    
@cython.boundscheck(False)
cdef void writePolygonSVG(Polygon * pol, FILE * f, bool contour, double cx, double cy) nogil:
  """helper function to write a sliced model to a SVG file in the style of slic3r --export-svg"""
  cdef size_t k
  #contour prefix
  fputs('    <polygon slic3r:type="', f)
  if contour:
    fputs('contour', f)
  else:
    fputs('hole', f)
  fputs('" points="', f)
  #points
  for k in range(pol[0].points.size()):
    if k>0:
      fputs(' ', f)
    fprintf(f, '%f,%f', pol[0].points[k].x*SCALING_FACTOR-cx, pol[0].points[k].y*SCALING_FACTOR-cy)
  #contour postfix
  fputs('" style="fill: ', f)
  if contour:
    fputs('white', f)
  else:
    fputs('black', f)
  fputs('" />\n', f)


#######################################################################
########## TRANSLATING from Clipper STRUCTURES TO SlicedModel##########
#######################################################################

@cython.boundscheck(False)
def ClipperObjects2SlicedModel(object clippers, cnp.ndarray[cnp.float64_t, ndim=1] zvalues):
  """Given a list or iterator of ClipperPolyTrees/ClipperPaths, and a
  concordant array of z values, create a SlicedModel. It is cheaper to
  convert ClipperPolyTrees than ClipperPaths"""
  cdef SlicedModel model = SlicedModel(zvalues)
  cdef size_t k
  model.thisptr[0].resize(zvalues.size)
  for k, obj in it.izip(xrange(zvalues.size), clippers):
    if   isinstance(obj, _c.ClipperPolyTree):
      tree = obj
      PolyTreeToExPolygons            ((<_c.ClipperPolyTree>obj).thisptr[0], &model.thisptr[0][k], False)
    elif isinstance(obj, _c.ClipperPaths):
      ClipperPaths_to_Slic3rExPolygons((<_c.ClipperPaths>   obj).thisptr[0], &model.thisptr[0][k], False)
    else:
      raise Exception('Invalid object type (neither ClipperPaths nor ClipperPolyTree)')
  return model

#######################################################################
########## TRANSLATING SlicedModel TO PYTHONIC STRUCTURE ##########
#######################################################################

cdef class ExPolygon:
  """Translation of ExPolygon to Python. Similar to a namedtuple"""
  cdef cnp.ndarray _contour
  cdef list        _holes

  property contour:
    def __get__(self):                  return self._contour
    def __set__(self, cnp.ndarray val): self._contour = val

  property holes:
    def __get__(self):                  return self._holes
    def __set__(self, list        val): self._holes = val
  
  def __cinit__(self, cnp.ndarray c=None, list hs=None, *args, **kwargs):
    self._contour = c
    self._holes   = hs
  
  #pickle protocol
  def __reduce__(self):
    d = {'_contour': self._contour, '_holes': self._holes}
    return (ExPolygon, (), d)
  def __setstate__(self, d):
    self._contour = d['_contour']
    self._holes   = d['_holes']
  
  def __str__(self):
    return "".join(("ExPolygon(contour=", self._contour.__str__(),  ", holes=", self._holes.__str__(),  ")"))
  def __repr__(self):
    return "".join(("ExPolygon(contour=", self._contour.__repr__(), ", holes=", self._holes.__repr__(), ")"))

cdef class Layer:
  """Translation of each one of the layers of a SlicedModel to Python. Similar to a namedtuple"""
  cdef double _z
  cdef list   _expolygons

  property z:
    def __get__(self):            return self._z
    def __set__(self,double val): self._z = val

  property expolygons:
    def __get__(self):            return self._expolygons
    def __set__(self, list  val): self._expolygons = val
  
  def __cinit__(self, double z=0.0, list exp=None, *args, **kwargs):
    self._z          = z
    self._expolygons = exp

  #pickle protocol
  def __reduce__(self):
    d = {'_z': self._z, '_exps': self._expolygons}
    return (Layer, (), d)
  def __setstate__(self, d):
    self._z          = d['_z']
    self._expolygons = d['_exps']
    
  def __len__(self):          return len(self._expolygons)
  def __iter__(self):         return self._slices.__iter__()
  def __getitem__(self, val): return self._expolygons.__getitem__(val)

  #Do not implement __setitem__, since this object's data may belong to a SlicedModel,
  #while we may manage to implement it, it is best to avoid unnecessary complexities 
  
  def __str__(self):
    return "".join(("Layer(z=", self._z.__str__(),  ", expolygons=", self._expolygons.__str__(),  ")"))
  def __repr__(self):
    return "".join(("Layer(z=", self._z.__repr__(), ", expolygons=", self._expolygons.__repr__(), ")"))

cdef class SliceCollection:
  """This class is the translation of SlicedModel to Python. It has been added
  in prevision that it might be needed in the future. It will be useful if more
  cdef attributes are added to SlicedModel (for example, if SCALING_FACTOR is
  un-hardcoded and becomes an attribute of SlicedModel)"""
  cdef list   _slices

  property slices:
    def __get__(self):          return self._slices
    def __set__(self,list val): self._slices = val

  def __cinit__(self, list slices=None, *args, **kwargs):
    self._slices = slices

  #pickle protocol
  def __reduce__(self):
    d = {'_slices': self._slices}
    return (SliceCollection, (), d)
    
  def __setstate__(self, d):    self._slices = d['_slices']
  def __len__(self):            return len(self._slices)
  def __iter__(self):           return self._slices.__iter__()
  def __getitem__(self, val):   return self._slices.__getitem__(val)

  #Do not implement __setitem__, since this object's data may belong to a SlicedModel,
  #while we may manage to implement it, it is best to avoid unnecessary complexities 
  
  def __str__(self):
    return "".join(("SliceCollection(", self._slices.__str__(),  ")"))
  def __repr__(self):
    return "".join(("SliceCollection(", self._slices.__repr__(), ")"))

  @cython.boundscheck(False)
  def toSlicedModel(self, SlicedModel model=None):
    """convert a SliceCollection back to a slicedModel.
    WARNING: the arrays must still be of type int64, otherwise the conversion will fail!"""
    cdef cnp.npy_intp length = len(self._slices)
    cdef size_t nlayer, nexp, nhole, npoint, lenexps, lenholes, lenpoints
    cdef Layer layer
    cdef list exps, holes
    cdef ExPolygon exp
    cdef Polygon *pol
    cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] array
    cdef cnp.ndarray zs = cnp.PyArray_EMPTY(1, &length, cnp.NPY_FLOAT64, 0)
    if model is None:
      model                       = SlicedModel(zs, True)
    else:
      model.thisptr[0].clear()
      model.zvalues               = zs
    model.thisptr[0].resize(length)
    #for each layer
    for nlayer in range(<size_t>length):
      layer                         = self._slices[nlayer]
      model.zvalues[nlayer]         = layer._z
      exps                          = layer._expolygons
      lenexps                       = len(exps)
      model.thisptr[0][nlayer].resize(lenexps)
      #for each expolygon
      for nexp in range(lenexps):
        exp                         = exps[nexp]
        array                       = exp._contour
        lenpoints                   = array.shape[0]
        pol                         = &model.thisptr[0][nlayer][nexp].contour
        pol[0].points.resize(lenpoints)
        #for each point in the contour
        for npoint in range(lenpoints):
          pol[0].points[npoint].x   = array[npoint,0]
          pol[0].points[npoint].y   = array[npoint,1]
        holes                       = exp._holes      
        lenholes                    = len(holes)
        model.thisptr[0][nlayer][nexp].holes.resize(lenholes)
        #for each hole
        for nhole in range(lenholes):
          array                     = holes[nhole]
          lenpoints                 = array.shape[0]
          pol                       = &model.thisptr[0][nlayer][nexp].holes[nhole]
          pol[0].points.resize(lenpoints)
          #for each point in the hole
          for npoint in range(lenpoints):
            pol[0].points[npoint].x = array[npoint,0]
            pol[0].points[npoint].y = array[npoint,1]
    return model
      
      

@cython.boundscheck(False)
cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] Polygon2arrayI(Polygon *pol):
  """helper function for ExPolygon2Tuple"""
  cdef vector[Point] points = pol[0].points
  cdef int sz = points.size()
  cdef int k
  cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] parr = np.empty((sz, 2), dtype=np.int64)
  #this may be wrapped with nogil, but it is probably not worth to do it so frequently
  for k in range(sz):
    parr[k,0] = points[k].x
    parr[k,1] = points[k].y
  return parr

@cython.boundscheck(False)
cdef cnp.ndarray[dtype=cnp.float64_t, ndim=2] Polygon2arrayF(Polygon *pol):
  """helper function for ExPolygon2Tuple"""
  cdef vector[Point] points = pol[0].points
  cdef size_t sz = points.size()
  cdef size_t k
  cdef cnp.ndarray[dtype=cnp.float64_t, ndim=2] parr = np.empty((sz, 2), dtype=np.float64)
  #this may be wrapped with nogil, but it is probably not worth to do it so frequently
  for k in range(sz):
    parr[k,0] = points[k].x*SCALING_FACTOR
    parr[k,1] = points[k].y*SCALING_FACTOR
  return parr

#strides have to be computed just once
cdef cnp.npy_intp *pointstrides = [sizeof(Point),
                                   <cnp.uint8_t*>&(<Point*>NULL).y - 
                                   <cnp.uint8_t*>&(<Point*>NULL).x]

cdef cnp.ndarray Polygon2arrayView(SlicedModel parent, Polygon *pol):
  """Similar to Polygon2arrayI, but instead of allocating a full-blown array,
  the returned array is a view into the underlying data"""
  cdef void         *data  = &(pol[0].points[0].x)
  cdef cnp.npy_intp *dims  = [pol[0].points.size(),2]
  cdef cnp.ndarray  result = cnp.PyArray_New(np.ndarray, 2, dims, cnp.NPY_INT64, pointstrides,
                                             data, -1, NPY_ARRAY_CARRAY, <object>NULL)
  ##result.base is of type PyObject*, so no reference counting with this assignment
  result.base              = <ref.PyObject*>parent
  ref.Py_INCREF(parent) #so, make sure that "result" owns a reference to "parent"
  #ref.Py_INCREF(result)
  return result
