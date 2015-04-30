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

from numpy.math cimport INFINITY#, NAN, isnan

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

cdef object slice2xrange(int size, rang):
  if rang is None:
    rango = xrange(size)
  elif isinstance(rang, slice):
    if rang.start is not None:
      if (not isinstance(rang.start, int)) or (rang.start<0) or (rang.start>=size):
        raise IndexError('Invalid start index range')
      if (not isinstance(rang.stop, int)) or (rang.stop<0) or (rang.stop>size):
        raise IndexError('Invalid stop range')
    rango = xrange(rang.start, rang.stop, rang.step)
  elif isinstance(rang, int):
    if (rang<0) or (rang>=size):
      raise IndexError('Invalid index')
    rango = (rang,)
  else:
    raise IndexError("Invalid indexing object")
  return rango
  

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

cdef class SlicedModel:
  """wrapper for the Slic3r data structure for a list of sliced layers"""
  
  property zvals:
    """expose the z values to python"""
    def __get__(self):
      return self.zvalues
    def __set__(self, cnp.ndarray[cnp.float32_t, ndim=1] val):
      self.zvalues = val

  def __cinit__(self, cnp.ndarray[cnp.float32_t, ndim=1] zvalues=None, bool doinit = True):
    if doinit:
      self.thisptr = new SLICEDMODEL()
    self.zvalues = zvalues

  def __dealloc__(self):
    del self.thisptr
  
  #quick and dirty (data is copied twice) pickle protocol
  #TODO: add support to load/save the data directly without intermediate structures like here
  def __reduce__(self):
    d = {'state': self.toLayerList(asView=True)}
    return (SlicedModel, (), d)
  def __setstate__(self, d):
    layerList2SlicedModel(d['state'], self)
    
  cdef bool slicesAreOrdered(self):
    return (np.diff(self.zvalues)>=0).all()
  
  #nogil SHOULD BE ALLOWED IN THE DEFINITIONS OF THE _removeXXX methods, BUT
  #WE CANNOT PUT IT BECAUSE OF A WEIRD CYTHON BUG (COMPILATION FAILS, COMPLAINING  
  #IN THE ARGUMENT LIST OF .remove(): "Converting to Python object not allowed without gil"
  
  @cython.boundscheck(False)  
  cdef void _removeLayers(self, unsigned int init, unsigned int end) nogil:
    cdef vector[ExPolygons].iterator it = self.thisptr[0].begin()
    self.thisptr[0].erase(it+init, it+end)
    
  @cython.boundscheck(False)  
  def removeLayers(self, unsigned int init, unsigned int end):
    """remove a range of layers"""
    _rangecheck(init, end, self.thisptr[0].size())
    self._remove(init, end)
    
  @cython.boundscheck(False)  
  cdef void _removeExPolygons(self, unsigned int nlayer, unsigned int init, unsigned int end) nogil:
    cdef vector[_ExPolygon].iterator it = self.thisptr[0][nlayer].begin()
    self.thisptr[0][nlayer].erase(it+init, it+end)
    
  @cython.boundscheck(False)  
  def removeExPolygons(self, unsigned int nlayer, unsigned int init, unsigned int end):
    """in a layer, remove a range of ExPolygons"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    _rangecheck(init, end, self.thisptr[0][nlayer].size())
    self._removeExPolygons(nlayer, init, end)
    
  @cython.boundscheck(False)  
  cdef void _removeHoles(self, unsigned int nlayer, unsigned int nexp, unsigned int init, unsigned int end) nogil:
    cdef vector[Polygon].iterator it = self.thisptr[0][nlayer][nexp].holes.begin()
    self.thisptr[0][nlayer][nexp].holes.erase(it+init, it+end)
    
  @cython.boundscheck(False)  
  def removeHoles(self, unsigned int nlayer, unsigned int nexp, unsigned int init, unsigned int end):
    """in an expolygon within a layer, remove a range of holes"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if nexp>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect ExPolygon ID')
    _rangecheck(init, end, self.thisptr[0][nlayer][nexp].holes.size())
    self._removeExPolygons(nlayer, init, end)
  
  @cython.boundscheck(False)  
  def select(self, cnp.ndarray[cnp.int64_t, ndim=1] selectedzs):
    """given an array of layer indexes, returns a new SlicedModel with a copy of
    those layers"""
    cdef SlicedModel selected = SlicedModel(self.zvalues[selectedzs])
    cdef bool ok = True
    cdef unsigned int k, siz
    siz = selectedzs.size
    with nogil:
      selected.thisptr[0].reserve(siz)
      for k in range(siz):
        if (selectedzs[k]>=0) and ((<unsigned int>selectedzs[k])<self.thisptr[0].size()):
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
      
  @cython.boundscheck(False)  
  def toLayerList(self, bool asInteger=False, bool asView=False, rang=None):
    """return a full-fledged pythonic representation of this SlicedModel object.
    
    This representation is a list of Layer objects.
    
    Each Layer object has a z value and a list of ExPolygon objects.
    
    An ExPolygon object represents a contour with holes. No nesting is allowed,
    i.e., there are no contours within the holes (however, nesting is implemented
    in practice as a list of logically unrelated but geometrically nested ExPolygons).
    
    Each ExPolygons has a contour and a list of holes.
    
    Contours and holes are numpy arrays of 2d points, each one representing a
    polygon."""
    
    rango = slice2xrange(self.thisptr[0].size(), rang)
    
    ret =  [Layer(self.zvalues[nlayer],
                  [ExPolygon(self.Polygon2array(&self.thisptr[0][nlayer][nexp].contour,      asInteger, asView),
                            [self.Polygon2array(&self.thisptr[0][nlayer][nexp].holes[nhole], asInteger, asView)
                              for nhole in xrange(self.numHoles(nlayer, nexp))])
                   for nexp in xrange(self.numExPolygons(nlayer))])
            for nlayer in rango]
    if isinstance(rango, tuple):
      return ret[0]
    return ret

  @cython.boundscheck(False)  
  def toExPolygonList(self, unsigned int nlayer, bool asInteger=False, bool asView=False, rang=None):
    """same as toLayerList(), but for ExPolygons"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    rango = slice2xrange(self.thisptr[0][nlayer].size(), rang)
    ret =  [ExPolygon(self.Polygon2array(&self.thisptr[0][nlayer][nexp].contour,      asInteger, asView),
                     [self.Polygon2array(&self.thisptr[0][nlayer][nexp].holes[nhole], asInteger, asView)
                              for nhole in xrange(self.numHoles(nlayer, nexp))])
                   for nexp in rango]
    if isinstance(rango, tuple):
      return ret[0]
    return ret

  def toHoleList(self, unsigned int nlayer, unsigned int nexp, bool asInteger=False, bool asView=False, rang=None):
    """same as toLayerList(), but for holes"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if   nexp>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect Expolygon ID')
    rango = slice2xrange(self.thisptr[0][nlayer][nexp].holes.size(), rang)
    ret =  [self.Polygon2array(&self.thisptr[0][nlayer][nexp].holes[nhole], asInteger, asView)
               for nhole in rango]
    if isinstance(rango, tuple):
      return ret[0]
    return ret
    
  def __len__(self):
    return self.thisptr[0].size()
  
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
        """
    asv = True
    if isinstance(val, int) or isinstance(val, slice):
      return self.toLayerList(asView=asv, rang=val)
    elif isinstance(val, tuple):
      ndims = len(val)
      if ndims==1:
        return self.toLayerList(asView=asv, rang=val[0])
      else:
        nlayer = val[0]
        if not isinstance(nlayer, int):
          raise IndexError('multidimensional slicing across layers is not supported')
        else:
          if ndims==2:
            if isinstance(val[1], basestring):
              return self.zvalues[nlayer]
            else:
              return self.toExPolygonList(nlayer, asView=asv, rang=val[1])
          else:
            nexp = val[1]
            if not isinstance(nexp, int):
              raise IndexError('multidimensional slicing across ExPolygons is not supported')
            else:
              if ndims>3:
                raise IndexError('ExPolygon components are arrays. If you want to slice them, do it separately')
              if isinstance(val[2], basestring): #get the contour
                return self.Polygon2array(&self.thisptr[0][nlayer][nexp].contour, False, asv)
              else:
                return self.toHoleList(nlayer, nexp, asView=True, rang=val[2])
    else:
      raise IndexError('Invalid slice object')
  
  @cython.boundscheck(False)  
  cpdef unsigned int numLayers(self):
    """number of layers of the sliced model"""
    return self.thisptr[0].size()
  
  @cython.boundscheck(False)  
  cpdef unsigned int numExPolygons(self, unsigned int nlayer):
    """number of ExPolygons in a layer of the sliced model"""
    if nlayer>=self.thisptr.size():
      raise IndexError('incorrect layer ID')
    return self.thisptr[0][nlayer].size()
    
  @cython.boundscheck(False)  
  cpdef unsigned int numHoles(self, unsigned int nlayer, unsigned int nExpolygon):
    """number of holes in an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect Expolygon ID')
    return self.thisptr[0][nlayer][nExpolygon].holes.size()

  @cython.boundscheck(False)
  def allExPolygons(self, bool asInteger=False, asView=False):
    """return a generator for all expolygons in all layers. Each ExPolygon is
    returned as a tuple with a layer index, an expolygon index (within the layer),
    a layer depth (z value), a contour and a list of holes. The contour and the
    holes are returned as numpy arrays, whose type is either numpy.int64 (native
    type of ExPolygon coordinates) or scaled numpy.float64 values"""
    cdef unsigned int k1, k2
    cdef double z
    cdef cnp.ndarray contour
    for k1 in xrange(self.thisptr[0].size()):
      z = self.zvalues[k1] #using yield, we cannot use a numpy array buffer declaration
      for k2 in xrange(self.thisptr[0][k1].size()):
        contour =  self.Polygon2array(&self.thisptr[0][k1][k2].contour,  asInteger, asView)
        holes   = [self.Polygon2array(&self.thisptr[0][k1][k2].holes[h], asInteger, asView)
                       for h in xrange(self.thisptr[0][k1][k2].holes.size())]
        yield (k1, k2, z, contour, holes)

  @cython.boundscheck(False)
  cdef cnp.ndarray Polygon2array(self, Polygon *pol, bool asInteger, bool asView):
    """common code for getting arrays from contours and holes"""
    if asView:
      return Polygon2arrayView(self, pol)
    else:
      if asInteger:
        return Polygon2arrayI(pol)
      else:
        return Polygon2arrayF(pol)*SCALING_FACTOR

  @cython.boundscheck(False)  
  def contour(self, unsigned int nlayer, unsigned int nExpolygon, bool asInteger=False, bool asView=False):
    """contour (as an array representing a list of points) of an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect Expolygon ID')
    return self.Polygon2array(&self.thisptr[0][nlayer][nExpolygon].contour, asInteger, asView)
    
  @cython.boundscheck(False)  
  def hole(self, unsigned int nlayer, unsigned int nExpolygon, unsigned int nhole, bool asInteger=False, bool asView=False):
    """hole (as an array representing a list of points) of an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise IndexError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise IndexError('incorrect Expolygon ID')
    if nhole>=self.thisptr[0][nlayer][nExpolygon].holes.size():
      raise IndexError('incorrect hole ID')
    return self.Polygon2array(&self.thisptr[0][nlayer][nExpolygon].holes[nhole], asInteger, asView)


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
  cdef unsigned int inew, nmodels, allsizes, reservesize, k, idxlowest, numToMerge
  cdef bool goon
  cdef double lowest, val
  cdef cnp.ndarray[cnp.float32_t, ndim=1] newzs
  cdef SLICEDMODEL *newptr = new SLICEDMODEL()
  cdef vector[int] numlayerss, idxs
  cdef vector[bool] ended, toMerge
  cdef vector[float] currentzs
  cdef vector[float*] zvaluess
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
    zvaluess[k]   = <float*>model.zvalues.data
    thisptrs[k]   = model.thisptr
    numlayerss[k] = model.thisptr[0].size()
    allsizes     += numlayerss[k]
    idxs[k]       = 0
    ended[k]      = numlayerss[k]==0

  newptr[0].reserve(allsizes)
  newzs     = np.empty((allsizes,), dtype=np.float32)
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
cdef void countPolygons(vector[vector[Polygons]] * polss, unsigned int *rnumP, unsigned int *rnumV) nogil:
  """compute the number of vertices and polygons in a vector of vectors of vectors of type Polygon"""
  cdef unsigned int numV = 0
  cdef unsigned int numP = 0
  cdef unsigned int k1, k2, k3
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
  cdef int k1, k2, nlayers, nexpols
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
  cdef unsigned int numP, numV, k1, k2, k3, k4, kp#, kt
  cdef cnp.ndarray[cnp.float32_t, ndim=1] zvalues = model.zvalues
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
  cdef unsigned int k1, k2, k3
  cdef double z, cx, cy, dx, dy, sx, sy
  cdef char space
  cdef FILE *f = fopen(filename, "w")
  cdef cnp.ndarray[cnp.float32_t, ndim=1] zvalues = model.zvalues
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
  cdef unsigned int numV, numP, numpoints, basecount, k1, k2, k3, k4
  cdef double z
  cdef FILE *f
  cdef unsigned int count = 0
  cdef cnp.ndarray[cnp.float32_t, ndim=1] zvalues = model.zvalues
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
  cdef unsigned int k1, k2, k3, k4
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
  cdef unsigned int k
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
########## TRANSLATING SlicedModel TO PYTHONIC STRUCTURE ##########
#######################################################################

cdef class ExPolygon:
  """Translation of ExPolygon to Python. Similar to a namedtuple"""
  cdef cnp.ndarray _contour
  cdef list        _holes

  property contour:
    def __get__(self):
      return self._contour
    def __set__(self, cnp.ndarray val):
      self.contour = val

  property holes:
    def __get__(self):
      return self._holes
    def __set__(self, list val):
      self.holes = val
  
  def __cinit__(self, cnp.ndarray c=None, list hs=None):
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
    def __get__(self):
      return self._z
    def __set__(self,double val):
      self.z = val

  property expolygons:
    def __get__(self):
      return self._expolygons
    def __set__(self, list val):
      self.expolygons = val
  
  def __cinit__(self, double z=0.0, list exp=None):
    self._z          = z
    self._expolygons = exp

  #pickle protocol
  def __reduce__(self):
    d = {'_z': self._z, '_exps': self._expolygons}
    return (Layer, (), d)
  def __setstate__(self, d):
    self._z          = d['_z']
    self._expolygons = d['_exps']
    
  def __len__(self):
    return len(self._expolygons)

  def __getitem__(self, val):
    return self._expolygons.__getitem__(val)

  #Do not implement __setitem__, since this object's data may belong to a SlicedModel,
  #while we may manage to implement it, it is best to avoid unnecessary complexities 
  
  def __str__(self):
    return "".join(("Layer(z=", self._z.__str__(),  ", expolygons=", self._expolygons.__str__(),  ")"))
  def __repr__(self):
    return "".join(("Layer(z=", self._z.__repr__(), ", expolygons=", self._expolygons.__repr__(), ")"))

@cython.boundscheck(False)
def layerList2SlicedModel(list layers, SlicedModel model=None):
  """convert a list of layers back to a slicedModel. WARNING: the arrays must still be of type int64"""
  cdef unsigned int length = len(layers)
  cdef unsigned int nlayer, nexp, nhole, npoint, lenexps, lenholes, lenpoints
  cdef Layer layer
  cdef list exps, holes
  cdef ExPolygon exp
  cdef Polygon *pol
  cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] array
  cdef cnp.ndarray zs = np.empty((length,), dtype=np.float32)
  if model is None:
    model                       = SlicedModel(zs, True)
  else:
    model.thisptr[0].clear()
    model.zvalues               = zs
  model.thisptr[0].resize(length)
  #for each layer
  for nlayer in range(length):
    layer                         = layers[nlayer]
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
  cdef int sz = points.size()
  cdef int k
  cdef cnp.ndarray[dtype=cnp.float64_t, ndim=2] parr = np.empty((sz, 2), dtype=np.float64)
  #this may be wrapped with nogil, but it is probably not worth to do it so frequently
  for k in range(sz):
    parr[k,0] = points[k].x
    parr[k,1] = points[k].y
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
