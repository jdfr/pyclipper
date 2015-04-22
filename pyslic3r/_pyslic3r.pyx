#cython: embedsignature=True


#while in development, we keep the libslic3rlib shared library in the very same
#directory as this file. In order to enable python to use this, execute this in
#the command line before entering python (it is necessary only once per bash session):
#    export LD_LIBRARY_PATH=.   <- or the relevant directory if we are not executing python from here
#more info: http://serverfault.com/questions/279068/cant-find-so-in-the-same-directory-as-the-executable

cimport cython
from libcpp cimport bool
from cpython cimport bool as boolp

cimport numpy as cnp
import numpy as np

#np.import_array()
cnp.import_array()


cimport slic3r_defs as sd
from slic3r_defs cimport *

from libc.stdio cimport *

cdef class SlicedModel:
  cdef vector[ExPolygons] *thisptr
  cdef cnp.ndarray zvalues
  
  property zvals:
    def __get__(self):
      return self.zvalues
    def __set__(self, cnp.ndarray[cnp.float32_t, ndim=1] val):
      self.zvalues = val
  
  def __cinit__(self, cnp.ndarray[cnp.float32_t, ndim=1] zvalues):
    self.thisptr = new vector[ExPolygons]()
    self.zvalues = zvalues

  def __dealloc__(self):
    del self.thisptr

#VERSION OF triangulateAllLayers for just one layer
#  @cython.boundscheck(False)  
#  cdef vector[Polygons] * triangulateLayer(self, unsigned int nlayer):
#    """generate a model of a layer apt to be represented in a 3D view"""
#    cdef int k, num
#    cdef vector[Polygons] * pols = new vector[Polygons]()
#    if nlayer>=self.thisptr.size():
#      raise ValueError('incorrect layer ID')
#    num = self.thisptr[0][nlayer].size()
#    pols[0].resize(num)
#    for k in range(num):
#      self.thisptr[0][nlayer][k].triangulate_pp(&pols[0][k])
#    return pols
  
  @cython.boundscheck(False)  
  cdef vector[vector[Polygons]] * triangulateAllLayers(self):
    """generate a model of the layers apt to be represented in a 3D view"""
    cdef int k1, k2, nlayers, nexpols
    cdef vector[vector[Polygons]] * polss = new vector[vector[Polygons]]()
    nlayers = self.thisptr[0].size()
    polss[0].resize(nlayers)
    for k1 in range(nlayers):
      nexpols = self.thisptr[0][k1].size()
      polss[0][k1].resize(nexpols)
      for k2 in range(nexpols):
        self.thisptr[0][k1][k2].triangulate_pp(&polss[0][k1][k2])
    return polss

  @cython.boundscheck(False)
  cdef tuple _layersAsTriangleMesh(self):
    """return a generator returning triangles"""
    cdef vector[vector[Polygons]] * polss
    cdef Points * polpoints
    cdef cnp.ndarray[cnp.float64_t, ndim=2] points
    cdef cnp.ndarray[cnp.int64_t, ndim=2] triangles
    cdef unsigned int k1, k2, k3, k4, kp#, kt
    cdef cnp.ndarray[cnp.float32_t, ndim=1] zvalues = self.zvalues
    cdef bool ok = True
    kp = 0
    #kt = 0
    polss = self.triangulateAllLayers()
    try:
      numP, numV = countPolygons(polss)
      points    = np.empty((numV, 3), dtype=np.float64)
      #triangles = np.empty((numP, 3), dtype=np.int64)
      triangles = np.arange(numV).reshape((-1, 3))
      for k1 in range(polss[0].size()):
        z = zvalues[k1]
        for k2 in range(polss[0][k1].size()):
          for k3 in range(polss[0][k1][k2].size()):
            polpoints = &polss[0][k1][k2][k3].points
            if polpoints[0].size()!=3:
              return (False, "Invalid triangulation!")
            for k4 in range(3):
              points[kp, 0]    = polpoints[0][k4].x*SCALING_FACTOR
              points[kp, 1]    = polpoints[0][k4].y*SCALING_FACTOR
              points[kp, 2]    = z
              #triangles[kt,k4] = kp
              kp += 1
            #kt += 1
      return (True, (points, triangles))
    finally:
      del polss

  @cython.boundscheck(False)
  def layersAsTriangleMesh(self):
    ret = self._layersAsTriangleMesh()
    if ret[0]:
      return ret[1]
    else:
      raise Exception(ret[1])
    
    
  
  @cython.boundscheck(False)  
  def writeAsSVG(self, filename):
    """write a SVG file in the style of slic3r --export-svg"""
    cdef unsigned int k1, k2, k3
    cdef double z, cx, cy, dx, dy, sx, sy
    cdef char space
    cdef FILE *f = fopen(filename, "w")
    cdef cnp.ndarray[cnp.float32_t, ndim=1] zvalues = self.zvalues
    cx, cy, dx, dy = self.computeBBParams()
    sx = cx-dx/2
    sy = cy-dy/2
    if f==NULL:
      raise ValueError('Could not open the file in write mode')
    try:
      #header
      fprintf(f, """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
<svg width="%f" height="%f" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:slic3r="http://slic3r.org/namespaces/slic3r">
<!-- 
Generated using pyslic3r pre-alpha
 -->
""", dx, dy)
      #layers
      for k1 in range(self.thisptr[0].size()):
        z = zvalues[k1]
        #layer prefix
        fprintf(f, '  <g id="layer%d" slic3r:z="%f">\n', k1, z)
        #expolygons
        for k2 in range(self.thisptr[0][k1].size()):
          #contour
          writePolygonSVG(&self.thisptr[0][k1][k2].contour, f, True, sx, sy)
          #holes
          for k3 in range(self.thisptr[0][k1][k2].holes.size()):
            writePolygonSVG(&self.thisptr[0][k1][k2].holes[k3], f, False, sx, sy)
        #layer postfix
        fputs('  </g>\n', f)
      #close svg
      fputs('</svg>', f)
    finally:
      fclose(f)
            
  @cython.boundscheck(False)
  cdef tuple computeBBParams(self):  
    """Compute some parameters of the bounding box: the center and the size"""
    cdef unsigned int k1, k2, k3, k4
    cdef double minx, maxx, miny, maxy, x, y, cx, cy, dx, dy
    
    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf
    for k1 in range(self.thisptr[0].size()):
      for k2 in range(self.thisptr[0][k1].size()):
        for k3 in range(self.thisptr[0][k1][k2].contour.points.size()):
          x = self.thisptr[0][k1][k2].contour.points[k3].x
          y = self.thisptr[0][k1][k2].contour.points[k3].y
          minx = min(minx, x)
          miny = min(miny, y)
          maxx = max(maxx, x)
          maxy = max(maxy, y)
        for k3 in range(self.thisptr[0][k1][k2].holes.size()):
          for k4 in range(self.thisptr[0][k1][k2].holes[k3].points.size()):
            x = self.thisptr[0][k1][k2].holes[k3].points[k4].x
            y = self.thisptr[0][k1][k2].holes[k3].points[k4].y
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
  def writeAsPLY(self, filename):
    """write a sliced model to a PLY file"""
    cdef vector[vector[Polygons]] * polss
    cdef unsigned int numV, numP, numpoints, basecount, k1, k2, k3, k4
    cdef double z
    cdef FILE *f
    cdef unsigned int count = 0
    cdef cnp.ndarray[cnp.float32_t, ndim=1] zvalues = self.zvalues
    polss = self.triangulateAllLayers()
    try:
      numP, numV = countPolygons(polss)
      f = fopen(filename, "w")
      if f==NULL:
        raise ValueError('Could not open the file in write mode')
      try:
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
  def toList(self):
    """convert into a list, each element is a layer, represented as a list of
    expolygons, where each expolygon is represented as a tuple (contour, holes),
    where "contour" is an array representing a sequence of 2D points,
    and "holes" is a list of arrays, each one representing a sequence of 2D points"""
    return layers2List(self.thisptr)
    
  @cython.boundscheck(False)  
  def numLayers(self):
    """number of layers of the sliced model"""
    return self.thisptr[0].size()
  
  @cython.boundscheck(False)  
  def numExPolygons(self, unsigned int nlayer):
    """number of ExPolygons in a layer of the sliced model"""
    if nlayer>=self.thisptr.size():
      raise ValueError('incorrect layer ID')
    return self.thisptr[0][nlayer].size()
    
  @cython.boundscheck(False)  
  def numHoles(self, unsigned int nlayer, unsigned int nExpolygon):
    """number of holes in an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise ValueError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise ValueError('incorrect Expolygon ID')
    return self.thisptr[0][nlayer][nExpolygon].holes.size()

#VERSION OF COUNTPOLYGONS FOR thisptr  
#  @cython.boundscheck(False)  
#  def countVertsAndPols(self):
#    """count the number of vertices and polygons (both contours and holes) in all layers"""
#    cdef unsigned int numV = 0
#    cdef unsigned int numP = 0
#    cdef unsigned int k1, k2, k3, v2, v3
#    for k1 in range(self.thisptr[0].size()):
#      v2 = self.thisptr[0][k1].size()
#      numP += v2
#      for k2 in range(v2):
#        v3 = self.thisptr[0][k1][k2].holes.size()
#        numP += v3
#        numV += self.thisptr[0][k1][k2].contour.points.size()
#        for k3 in range(v3):
#          numV += self.thisptr[0][k1][k2].holes[k3].points.size()
#    return (numV, numP)

#VERSION OF allExPolygons FOR JUST ONE LAYER
#  @cython.boundscheck(False)  
#  def layerExPolygons(self, unsigned int nlayer, bool asInteger=False):
#    """return a generator for all expolygons in a layer. Each ExPolygon is returned as a tuple
#    with an expolygon index (within the layer), a contour and a list of holes"""
#    cdef unsigned int k
#    cdef cnp.ndarray contour
#    if nlayer>=self.thisptr[0].size():
#      raise ValueError('incorrect layer ID')
#    for k in xrange(self.thisptr[0][nlayer].size()):
#      contour = self._contour(nlayer, k, asInteger)
#      holes = [self._hole(nlayer, k, h, asInteger) for h in xrange(self.thisptr[0][nlayer][k].holes.size())]
#      yield (contour, holes)
      
  @cython.boundscheck(False)
  def allExPolygons(self, bool asInteger=False):
    """return a generator for all expolygons in all layers. Each ExPolygon is returned as a tuple
    with a layer index, an expolygon index (within the layer), a layer depth (z value), a contour and a list of holes"""
    cdef unsigned int k1, k2
    cdef double z
    cdef cnp.ndarray contour
    for k1 in xrange(self.thisptr[0].size()):
      z = self.zvalues[k1] #using yield, we cannot use the 
      for k2 in xrange(self.thisptr[0][k1].size()):
        contour = self._contour(k1, k2, asInteger)
        holes = [self._hole(k1, k2, h, asInteger) for h in xrange(self.thisptr[0][k1][k2].holes.size())]
        yield (k1, k2, z, contour, holes)

  @cython.boundscheck(False)  
  cdef cnp.ndarray _contour(self, unsigned int nlayer, unsigned int nExpolygon, bool asInteger):
    "get a contour as an array containing a list of points"""
    if asInteger:
      return Polygon2arrayI(&self.thisptr[0][nlayer][nExpolygon].contour)
    else:
      return Polygon2arrayF(&self.thisptr[0][nlayer][nExpolygon].contour)*SCALING_FACTOR

  @cython.boundscheck(False)  
  cdef cnp.ndarray _hole(self, unsigned int nlayer, unsigned int nExpolygon, unsigned int nhole, bool asInteger):
    "get a hole as an array containing a list of points"""
    if asInteger:
      return Polygon2arrayI(&self.thisptr[0][nlayer][nExpolygon].holes[nhole])
    else:
      return Polygon2arrayF(&self.thisptr[0][nlayer][nExpolygon].holes[nhole])*SCALING_FACTOR
      
  @cython.boundscheck(False)  
  def contour(self, unsigned int nlayer, unsigned int nExpolygon, bool asInteger=False):
    """contour (as an array representing a list of points) of an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise ValueError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise ValueError('incorrect Expolygon ID')
    return self._contour(nlayer, nExpolygon, asInteger)
    
  @cython.boundscheck(False)  
  def hole(self, unsigned int nlayer, unsigned int nExpolygon, unsigned int nhole, bool asInteger=False):
    """hole (as an array representing a list of points) of an ExPolygon of a layer of the sliced model"""
    if nlayer>=self.thisptr[0].size():
      raise ValueError('incorrect layer ID')
    if nExpolygon>=self.thisptr[0][nlayer].size():
      raise ValueError('incorrect Expolygon ID')
    if nhole>=self.thisptr[0][nlayer][nExpolygon].holes.size():
      raise ValueError('incorrect hole ID')
    return self._hole(nlayer, nExpolygon, nhole, asInteger)
    

@cython.boundscheck(False)
cdef tuple countPolygons(vector[vector[Polygons]] * polss):
  """compute the number of vertices and polygons in a vector of vectors of vectors of type Polygon"""
  cdef unsigned int numV = 0
  cdef unsigned int numP = 0
  cdef unsigned int k1, k2, k3
  for k1 in range(polss[0].size()):
    for k2 in range(polss[0][k1].size()):
      for k3 in range(polss[0][k1][k2].size()):
        numP += 1
        numV += polss[0][k1][k2][k3].points.size()
  return (numP, numV)


def doTest(SlicedModel sliced):
  cdef vector[ExPolygons] *otro
  
  otro = new vector[ExPolygons]()
  otro[0] = sliced.thisptr[0]
  print otro[0][0][0].contour.points[0].x
  print sliced.thisptr[0][0][0].contour.points[0].x
  otro[0][0][0].contour.points[0].x = 350
  print otro[0][0][0].contour.points[0].x
  print sliced.thisptr[0][0][0].contour.points[0].x
  del otro
  
cdef class SlicerMesh:
  """class to represent a STL mesh and slice it"""
  cdef sd.TriangleMesh *thisptr
  
  def __cinit__(self, filename):
    self.thisptr = new sd.TriangleMesh()
    self.thisptr.ReadSTLFile(filename)
  def __dealloc__(self):
    del self.thisptr

  @cython.boundscheck(False)
  def doslice(self, cnp.ndarray[cnp.float32_t, ndim=1] zs):
    cdef int k, k1, sz, sz1
    cdef vector[float] zsv
    cdef TriangleMeshSlicer *slicer = new TriangleMeshSlicer(self.thisptr)
    cdef SlicedModel layers = SlicedModel(zs.copy())
    try:
      sz = zs.size
      zsv.resize(sz)
      for k in range(sz):
        zsv[k] = zs[k]
      slicer.slice(zsv, layers.thisptr)
      return layers
    finally:
      del slicer


@cython.boundscheck(False)
cdef void writePolygonSVG(Polygon * pol, FILE * f, bool contour, double cx, double cy):
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

@cython.boundscheck(False)
cdef list layers2List(vector[ExPolygons] *layers):
  """helper function to transform a sliced model into a native Python structure"""
  cdef int sz = layers[0].size()
  cdef int k
  lays = [ExPolygons2List(&layers[0][k]) for k in range(sz)]
#  lays = [None]**sz
#  for k in range(sz):
#    lays[k] = ExPolygons2List(&layers[0][k])
  return lays

@cython.boundscheck(False)
cdef list ExPolygons2List(vector[ExPolygon] *expols):
  """helper function for layers2List"""
  cdef int sz = expols[0].size()
  cdef int k
  pols = [ExPolygon2Tuple(&expols[0][k]) for k in range(sz)]
#  pols = [None]**sz
#  for k in range(sz):
#    pols[k] = ExPolygon2Tuple(&expols[0][k])
  return pols
      
@cython.boundscheck(False)
cdef tuple ExPolygon2Tuple(ExPolygon *expol):
  """helper function for ExPolygons2List"""
  cdef Polygon *pol
  cdef int k, sz
  pol = &expol.contour
  cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] contour = Polygon2arrayI(&expol[0].contour)
#  cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] hole
  sz = expol[0].holes.size()
  holes = [Polygon2arrayI(&expol[0].holes[k]) for k in range(sz)]
  return (contour, holes)

@cython.boundscheck(False)
cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] Polygon2arrayI(Polygon *pol):
  """helper function for ExPolygon2Tuple"""
  cdef vector[Point] points = pol[0].points
  cdef int sz = points.size()
  cdef int k
  cdef cnp.ndarray[dtype=cnp.int64_t, ndim=2] parr = np.empty((sz, 2), dtype=np.int64)
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
  for k in range(sz):
    parr[k,0] = points[k].x
    parr[k,1] = points[k].y
  return parr
  
    

#def stl2obj(basestring fin, basestring fout):
#  cdef TriangleMesh mesh
#  mesh.ReadSTLFile(fin)
#  #mesh.WriteOBJFile(fout)
#  mesh.write_binary(fout)


#import pyslic3r as s
#s.stl2obj("/home/josedavid/3dprint/software/slicers/multi/nested001.stl", "/home/josedavid/3dprint/software/slicers/multi/otro.stl")