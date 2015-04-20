#cython: embedsignature=True


#while in development, we keep the libslic3rlib shared library in the very same
#directory as this file. In order to enable python to use this, execute this in
#the command line before entering python (it is necessary only once per bash session):
#    export LD_LIBRARY_PATH=.   <- or the relevant directory if we are not executing python from here
#more info: http://serverfault.com/questions/279068/cant-find-so-in-the-same-directory-as-the-executable


from libcpp cimport bool
from cpython cimport bool as boolp

#cimport numpy as np
#import numpy as np

#np.import_array()


cimport slic3r_defs as s

def stl2obj(basestring fin, basestring fout):
  print "HOLA 1"
  cdef s.TriangleMesh mesh
  print "HOLA 2"
  mesh.ReadSTLFile(fin)
  print "HOLA 3"
  #mesh.WriteOBJFile(fout)
  mesh.write_binary(fout)
  print "HOLA 4"


#import pyslic3r as s
#s.stl2obj("/home/josedavid/3dprint/software/slicers/multi/nested001.stl", "/home/josedavid/3dprint/software/slicers/multi/otro.stl")