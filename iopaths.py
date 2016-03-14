import pyclipper.Clipper as clipper
import numpy as n

from collections import namedtuple

TYPE_RAW_CONTOUR       = 0
TYPE_PROCESSED_CONTOUR = 1
TYPE_TOOLPATH          = 2
ALL_TYPES              = [TYPE_RAW_CONTOUR, TYPE_PROCESSED_CONTOUR, TYPE_TOOLPATH]

SAVEMODE_INT64         = 0
SAVEMODE_DOUBLE        = 1
SAVEMODE_DOUBLE_3D     = 2

class PathsRecord:
  def __init__(self, typ=None, ntool=None, z=None, savemode=None, scaling=None, paths=None):
    self.numbytes   = -1
    self.headersize = -1
    self.type       = typ
    self.ntool      = ntool
    self.z          = z
    self.savemode   = savemode
    self.scaling    = scaling
    self.paths      = paths
    
  def readFromFile(self, f):
    header = (f.readInt64(), f.readInt64(), f.readInt64(), f.readInt64(), f.readDouble(), f.readInt64(), f.readDouble())
    self.numbytes, self.headersize, self.type, self.ntool, self.z, self.savemode, self.scaling = header
    #print header
    for ii in xrange((self.headersize-len(header)*8)/8):
      dummy   = f.readInt64()
    if   self.savemode==SAVEMODE_INT64:
      self.paths = clipper.ClipperPaths()
      self.paths.fromStream(f)
    elif self.savemode==SAVEMODE_DOUBLE:
      self.paths = clipper.ClipperDPaths()
      self.paths.fromStream(f)
    elif self.savemode==SAVEMODE_DOUBLE_3D:
      self.paths = clipper.read3DDoublePathsFromFile(f)
    else:
      raise Exception("save format was declared as %d, but this value is not recognized!!!" % self.savemode)
  
  def writeToFile(self, f):
    isclipper = isinstance(self.paths, clipper.ClipperPaths)
    isdouble  = isinstance(self.paths, clipper.ClipperDPaths)
    is3d      = isinstance(self.paths, (list, tuple)) and all([isinstance(p, n.ndarray) and len(p.shape)==2 and p.shape[1]==3 for p in self.paths])
    if is3d:
      raise Exception('Saving 3D paths is not supported for now')
    if isclipper or isdouble or is3d:
      siz = (1+len(self.paths)+sum(p.size for p in self.paths))*8
    else:
      raise Exception('type of paths (%s) is not writable!!!' % str(type(self.paths)))
    header = [-1, -1, self.type, self.ntool, self.z, self.savemode, self.scaling]
    header[1] = len(header)*8
    header[0] = header[1]+siz
    for h in header:
      f.write(h)
    if isclipper or isdouble:
      self.paths.toStream(f)
    elif is3d:
      raise Exception('Saving 3D paths is not supported for now')

class FileContents:
  def __init__(self):
    self.magic        = 1213481296 #struct.unpack('I','PATH')==1213481296
    self.version      = 0
    self.numtools     = 0
    self.usezradiuses = False
    self.xradiuses    = []
    self.zradiuses    = []
    self.zheights     = []
    self.zapppoints   = []
    self.numpaths     = 0
    self.records      = []
    self.zs           = []
  
  def readFromFile(self, filename):
    f = clipper.File(filename, 'rb', False)
    self.magic        = f.readInt32()
    self.version      = f.readInt32()
    self.numtools     = f.readInt64()
    self.usezradiuses = f.readInt64()!=0
    self.xradiuses    = [None]*self.numtools
    self.zradiuses    = [None]*self.numtools
    self.zheights     = [None]*self.numtools
    self.zapppoints   = [None]*self.numtools
    for idx in xrange(self.numtools):
      self.xradiuses   [idx] = f.readDouble()
      if self.usezradiuses:
        self.zradiuses [idx] = f.readDouble()
        self.zheights  [idx] = f.readDouble()
        self.zapppoints[idx] = f.readDouble()
    
    self.numpaths     = f.readInt64()
    self.records      = [None] * self.numpaths
    allzs            = set()
    for i in xrange(self.numpaths):
      value = PathsRecord()
      try:
        value.readFromFile(f)
      except Exception as e:
        v = str(e.args[0]) if len(e.args)==1 else str(e.args)
        raise Exception("While reading record %d of file %s: %s" % (i, filename if not filename is None else "standard input", v))
      self.records[i]  = value
      allzs.add(value.z)
    f.close()  
    
    self.zs = sorted(list(allzs))

  def organizeRecords(self):
    if type(self.records)==dict:
      return
    pathsbytype = dict()
    for value in self.records:
      key1      = (value.type, value.ntool)
      key2      = value.z
      if not key1 in pathsbytype:
        pathsbytype[key1] = {key2:value}
      elif not key2 in pathsbytype[key1]:
        pathsbytype[key1][key2] = value
      else:
        raise Exception('repeated key combo type=%d, ntool=%d, z=%f' % (value.type, value.ntool, value.z))
        #pathsbytype[key1][key2].append(value)
    self.records = pathsbytype

  def writeToFile(self, filename):
    numpaths = int(len(self.records))
    usez     = self.usezradiuses
    f        = clipper.File(filename, 'wb', True)
    header   = self.magic*(2**32)+self.version
    f.writeInt32(self.magic)
    f.writeInt32(self.version)
    f.write(self.numtools)
    f.write(self.usezradiuses)
    for idx in xrange(self.numtools):
      f.write(self.xradiuses[idx])
      if usez:
        f.write(self.zradiuses[idx])
        f.write(self.zheights[idx])
        f.write(self.zapppoints[idx])
    f.write(numpaths)
    for idx in xrange(numpaths):
      try:
        self.records[idx].writeToFile(f)
      except Exception as e:
        v = str(e.args[0]) if len(e.args)==1 else str(e.args)
        raise Exception("While writing record %d of file %s: %s" % (idx, filename if not filename is None else "standard output", v))
    f
