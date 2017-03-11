import stl
import sys
import numpy as n

if __name__ == "__main__":
  if len(sys.argv)<4:
    sys.stderr.write('I need at least two stl input files and a output file name!\n')
    sys.exit()

  stls = []
  addargs = dict(update_normals=False, remove_empty_areas=False, unpack_data=False)
  for i in range(1, len(sys.argv)-1):
    stlf = stl.StlMesh(sys.argv[i], **addargs)
    stls.append(stlf.data)
  
  alldata = n.concatenate(tuple(stls))
  allstl = stl.StlMesh('', data=alldata, **addargs)
  allstl.save(sys.argv[-1], mode=stl.BINARY, update_normals=False)


