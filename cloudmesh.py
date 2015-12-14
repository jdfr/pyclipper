import numpy as n
#from scipy.interpolate import SmoothBivariateSpline
from scipy.spatial import Delaunay
import stl
import sys
import traceback
from collections import namedtuple
from cmdutils import readStringParam, readFloatParam

RetVal = namedtuple('RetVal', ['ok', 'val'])

def usage():
  print "arguments: CLOUDFILEIN SEPARATOR ZMIN ZMAX ZSUB STLFILEOUT"
  print "           CLOUDFILEIN: point cloud input file (will be interpreted as a heightmap)"
  print "           SEPARATOR: the separator used in CLOUDFILEIN. It can be any character."
  print "                      Use quotes if the character is a space or it is interpreted"
  print "                      by the command line interpreter (ex: \" \", or \";\")"
  print "           ZMIN, ZMAX: points from the point cloud will be removed if they are"
  print "                       outside the open range (ZMIN, ZMAX)"
  print "           ZSUB: The output file will be based on the points from the point cloud"
  print "                 plus a base of thickness ZSUB"
  print "           STLFILEOUT: STL output file"
  sys.exit(-1)

def main(argv):
  argidx = 1
  
  inputfile,  argidx = readStringParam(argv, argidx, usage, "NO ARGUMENTS!\n")
  separator,  argidx = readStringParam(argv, argidx, usage, "NO SEPARATOR!\n")
  if len(separator)!=1:
    print "SEPARATOR MUST BE A SINGLE CHARACTER!\n"
    usage()
  zmin,       argidx =  readFloatParam(argv, argidx, usage, "ZMIN")
  zmax,       argidx =  readFloatParam(argv, argidx, usage, "ZMAX")
  zsub,       argidx =  readFloatParam(argv, argidx, usage, "ZSUB")
  outputfile, argidx = readStringParam(argv, argidx, usage, "NO OUTPUT STL FILE!\n")
  
  try:
    cloud = n.loadtxt(inputfile, delimiter=separator)
  except:
    print "Could not open file <%s> as a XYZ point cloud with delimiter <%s>" % (inputfile, separator)
    sys.exit(-1)
  
  if len(cloud.shape)!=2:
    print "file %s does not contain a XYZ point cloud"
    sys.exit(-1)
  
  if cloud.shape[1]<3:
    print "file %s does not contain a XYZ point cloud (only %d coordinates)" % cloud.shape[1]
    sys.exit(-1)
  
  if cloud.shape[1]>3:
    cloud = cloud[:,0:3]
  
  cloud = preprocessPointCloud(cloud, [zmin, zmax])
  
  try:
    result = createMeshFromPointCloud(cloud, zsub)
    invertfaces = True
  except:
    print "Unexpected exception:"
    traceback.print_exc()
    sys.exit(-1)
  
  del cloud
  
  if not result.ok:
    print result.val
    sys.exit(-1)
  
  points, faces = result.val
  
  try:
    stlobj = triangleMeshToStlObject(points, faces, invertfaces)
    
    del result, points, faces
    
    stlobj.save(outputfile, mode=stl.BINARY, update_normals=False)
  except:
    print "Unexpected exception:"
    traceback.print_exc()
    sys.exit(-1)
  
def preprocessPointCloud(points, zlimits):
  """remove invalid points"""
  if len(zlimits)==2:
    mask   = n.logical_and(points[:,2]<zlimits[1], points[:,2]>zlimits[0])
    if mask.all():
      usedPoints = points
    else:
      usedPoints = points[mask,:]
  else:
    usedPoints = points
  return usedPoints

def triangleMeshToStlObject(points, triangles, invertfaces):
  data = n.zeros((triangles.shape[0],), dtype=stl.Mesh.dtype)
  #numpy's expressive power to the rescue!
  if invertfaces:
    data['vectors'] = points[triangles[:,::-1],:]
  else:
    data['vectors'] = points[triangles,:]
  return stl.StlMesh("", update_normals=False, data=data)
  
def meshForGrid(shape):
  """create a mesh for a grid in C-style matrix order"""
  idxs = n.arange(n.prod(shape)).reshape(shape)
  #v1, v2, v3, v4: vertices of each square in the grid, clockwise
  v1 = idxs[:-1,:-1]
  v2 = idxs[:-1,1:]
  v3 = idxs[1:,1:]
  v4 = idxs[1:,:-1]
  faces = n.vstack((
      n.column_stack((v1.ravel(), v2.ravel(), v4.ravel())),    #triangles type 1
      n.column_stack((v2.ravel(), v3.ravel(), v4.ravel())) ))  #triangles type 2
  return faces
  
def createMeshFromResampledPointCloud(points, zsub, step):
  """Given a point cloud interpreted as a heightmap,
resample said heightmap in its bounding box, and create a mesh"""
  raise Exception('Not implemented yet')
  minx = points[:,0].min()
  maxx = points[:,0].max()
  miny = points[:,1].min()
  maxy = points[:,1].max()
  minz = points[:,2].min()
  spline = SmoothBivariateSpline(points[:,0], points[:,1], points[:,2], bbox=[minx, maxx, miny, maxy])
  newx = n.arange(minx, maxx+step,step)
  newy = n.arange(miny, maxy+step,step)
  facesU = meshForGrid((newx.size, newy.size))
  dx = 0
  dy = 0
  newz = spline(newx, newy, dx=dx, dy=dy, grid=True)
  newx, newy = n.meshgrid(newx, newy)
  #TODO: get border points, then apply the same algorithm as in createMeshFromPointCloud()
  raise Exception('Not implemented yet')
  
  
  
def createMeshFromPointCloud(points, zsub):
  """Given a point cloud interpreted as a heightmap,
creates a mesh from said point cloud, as a cylinder:
top and base meshes connected by a lateral ribbon"""
  #create upper face
  try:
    #tessU = Delaunay(points[:,0:2], qhull_options='QJ') #This is to make sure that all points are used
    tessU = Delaunay(points[:,0:2])
  except:
    traceback.print_exc()
    return RetVal(False, 'Error trying to generate a mesh from the point cloud: Delaunay triangulation of the point cloud failed:\n'+traceback.format_exc())
  tU = tessU.simplices
  
  #get border edges in border triangles
  i1, i2 = (tessU.neighbors==-1).nonzero() #indexes of vertexes not opossed to a triangle, they are not in the edge of the mesh, but the other two vertexes of the triangle are!
  i21 = (i2+1)%3  #these are the column indexes of vertexes in the edge of the mesh
  i22 = (i21+1)%3 #
  ps = n.column_stack((tU[i1,i21], tU[i1,i22])) #edges at the edge of the mesh
  #order the points in the edges (counterclockwise)
  ordered = n.empty(ps.shape[0], dtype=n.int32)
  ordered[0] = ps[0,0] #seed the sequence with the first edge
  ordered[1] = ps[0,1]
  ps[0,:] = -1
  io = 2
  while io<ordered.size:
    i1, i2 = (ps==ordered[io-1]).nonzero() #get the position of the last vertex in the ordered sequence
    if i1.size!=1: #each vertex should appear only twice in the list of edges
      return RetVal(False, "Error: could not get ordered border for delaunay triangulation")
    ordered[io] = ps[i1, (i2+1)%2] #add the adjacent vertex to the ordered list
    ps[i1,:] = -1 #remove the edge from the list of edges
    io += 1
  #points in the base are those at the edge, but lowered by a certain amount  
  newpoints = points[ordered,:]
  if zsub>0:
    newpoints[:,2] = points[:,2].min()-zsub
  else:
    newpoints[:,2] = points[:,2].max()-zsub
  #ordered list of vertexes at the edges of the upper mesh
  nidxU = ordered
  #same, list, but shifted
  nidxUp1 = n.concatenate((ordered[ordered.shape[0]-1:ordered.shape[0]], ordered[0:-1]))
  #ordered list of vertexes at the lower mesh
  nidxL = n.arange(points.shape[0], points.shape[0]+newpoints.shape[0])
  #same, list, but shifted the other way around
  nidxLm1 = n.concatenate((nidxL[1:nidxL.size], nidxL[0:1]))
  #triangles for the connecting ribbon
  Tmed1 = n.column_stack((nidxU, nidxUp1, nidxL))
  Tmed2 = n.column_stack((nidxLm1, nidxU, nidxL))
  #get base mesh
  try:
    #tessB = Delaunay(newpoints[:,0:2], qhull_options='QJ') #This is to make sure that all points are used
    tessB = Delaunay(newpoints[:,0:2])
  except:
    traceback.print_exc()
    return RetVal(False, 'Error trying to generate a mesh from the point cloud: Delaunay triangulation of the base failed:\n'+traceback.format_exc())
  #reindex the triangles of the base mesh
  tB = nidxL[tessB.simplices]
  # this is to have all triangles of the base mesh to be counterclockwise
  tB = tB[:,[0,2,1]] 
  tA = n.concatenate((tU, Tmed1, Tmed2, tB))
  #create arrays with all points
  allPoints  = n.concatenate((points, newpoints))
  return RetVal(True, (allPoints, tA))

if __name__=='__main__':
  main(sys.argv)