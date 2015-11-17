import pyclipper.Clipper as clipper

from collections import namedtuple

TYPE_RAW_CONTOUR       = 0
TYPE_PROCESSED_CONTOUR = 1
TYPE_TOOLPATH          = 2
ALL_TYPES              = [TYPE_RAW_CONTOUR, TYPE_PROCESSED_CONTOUR, TYPE_TOOLPATH]

SAVEMODE_INT64         = 0
SAVEMODE_DOUBLE        = 1
SAVEMODE_DOUBLE_3D     = 2


InputPath = namedtuple('InputPath', ['type', 'savemode', 'ntool', 'z', 'paths', 'scaling'])
FileContents = namedtuple('FileContents', ['numtools', 'usezradiuses', 'xradiuses', 'zradiuses', 'numpaths', 'paths', 'zs'])

def readFile(filename):
  f = clipper.File(filename, 'rb', False)

  #get paths by z
  numtools    = f.readInt64()
  useZRadiuses = f.readInt64()!=0
  xradiuses   = [None]*numtools
  zradiuses   = [None]*numtools
  for idx in xrange(numtools):
    xradiuses[idx] = f.readDouble()
    if useZRadiuses:
      zradiuses[idx] = f.readDouble()
  numpaths    = f.readInt64()
  #dictionary of dictionaries: the first key is (type, ntool), the second is z
  paths       = [None] * numpaths
  allzs       = set()
  for i in xrange(numpaths):
    header = (f.readInt64(), f.readInt64(), f.readInt64(), f.readInt64(), f.readDouble(), f.readInt64(), f.readDouble())
    numbytes, headersiz, typ, ntool, z, savemode, scaling = header
    #print header
    for ii in xrange(headersiz-len(header)*8):
      dummy   = f.readInt64()
    if   savemode==SAVEMODE_INT64:
      dpaths  = clipper.ClipperPaths()
      dpaths.fromStream(f)
    elif savemode==SAVEMODE_DOUBLE:
      dpaths  = clipper.ClipperDPaths()
      dpaths.fromStream(f)
    elif savemode==SAVEMODE_DOUBLE_3D:
      dpaths = clipper.read3DDoublePathsFromFile(f)
    else:
      raise Exception("While reading file %s, save format of %d-th paths is %d, but this value is not recognized!!!" % (filename if not filename is None else "standard input", i, savemode))
    value     = InputPath(typ, savemode, ntool, z, dpaths, scaling)
    paths[i]  = value
    key1      = (typ, ntool)
    key2      = z
    allzs.add(z)
  
  f.close()  
  
  allzs = sorted(list(allzs))
  return FileContents(numtools=numtools, usezradiuses=useZRadiuses, xradiuses=xradiuses, zradiuses=zradiuses, numpaths=numpaths, paths=paths, zs=allzs)

def organizePaths(contents):
  if type(contents.paths)==dict:
    return contents
  pathsbytype = dict()
  for value in contents.paths:
    key1      = (value.type, value.ntool)
    key2      = value.z
    if not key1 in pathsbytype:
      pathsbytype[key1] = {key2:value}
    elif not key2 in pathsbytype[key1]:
      pathsbytype[key1][key2] = value
    else:
      raise Exception('repeated key combo type=%d, ntool=%d, z=%f' % (typ, ntool, z))
      #pathsbytype[key1][key2].append(value)
  return FileContents(numtools=contents.numtools, usezradiuses=contents.usezradiuses, xradiuses=contents.xradiuses, zradiuses=contents.zradiuses, numpaths=contents.numpaths, paths=pathsbytype, zs=contents.zs)
