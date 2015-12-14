import numpy as n

import sys

import matplotlib.pyplot as plt

#This class computes the contact angle for the model in:
#   "Wetting in 1+1 dimensions with two-scale roughness",
#   Joel de Coninck, Francois Dunlop and Thierry Huillet
#   Physica A: Statistical Mechanics and its Applications,
#   2015, vol. 438, issue C, pages 398-415
class Theta:
  def __init__(self, a2, b2, c2, k1, k2, b1, computeAllParams=True):
    """Parameters:
          a2: width  of rectangular protusions at the second (small) scale
          b2: height of rectangular protusions at the second (small) scale
          c2: width  of space between rectangular protusions at the second (small) scale
          k1: number of small rectangular protusions in a big-scale rectangular protusion
          k2: number of small rectangular protusions between two big-scale rectangular protusions
          b1: height of rectangular protusions at the first (big) scale
    """
    self.a2    = a2
    self.b2    = b2
    self.c2    = c2
    self.k1    = k1
    self.k2    = k2
    self.b1    = b1
    
    self.a1    = self.k1*self.a2+(self.k1-1)*self.c2
    self.c1    = self.k2*self.a2+(self.k2+1)*self.c2
    
    self.printfun = sys.stdout.write
    
    if computeAllParams:
      computeParams()
      
  def computeParams(self):
    self.b     = self.b1/(self.k1+self.k2)
    
    self.rho   = self.k1/self.k2
    
    self.phi1  = self.rho/(1+self.rho)
    self.phi2  = self.a2/(self.a2+self.c2)
    
    self.r1    = 1+(2*self.b/(self.a2+self.c2))
    self.r2    = 1+(2*self.b2/(self.a2+self.c2))
    
    self.params = ['a2', 'b2', 'c2', 'k1', 'k2', 'a1', 'b1', 'c2', 'b', 'rho', 'phi1', 'phi2', 'r1', 'r2']
    
    self.funcs = [self.costh_CB12, self.costh_CW1CB2, self.costh_W1CB2, self.costh_W12, self.costh_CB1W2, self.costh_W1W2]
    self.names = [          'CB12',          'CW1CB2',          'W1CB2',          'W12',          'CB1W2',          'W1W2']
    self.cols  = [          'r',             'g',               'b',              'c',            'm',              'k']
    
    H          = (2*self.b2)>(self.c2*n.sqrt(2)-1)
    if not H:
      raise ValueError('Condition H is not met!!!') 
    
  def printparams(self):
    for p in self.params:
      self.printfun("%s = %f\n" % (p, getattr(self, p)))
    
  def costh2_CB   (self, cos_th0): return self.phi2*(1+cos_th0)-1
  def costh2_W    (self, cos_th0): return self.r2*cos_th0
  def sinth2_CB   (self, cos_th0): return n.sin(n.arccos(self.costh2_CB(cos_th0)))
  def sinth2_W    (self, cos_th0): return n.sin(n.arccos(self.costh2_W (cos_th0)))
    
  def costh_CB12  (self, cos_th0): return   self.phi1*self.phi2*(cos_th0+1)-1
  def costh_CW1CB2(self, cos_th0): return -(self.r1        -1)*self.sinth2_CB(cos_th0)+self.costh2_CB(cos_th0)
  def costh_W1CB2 (self, cos_th0): return  (self.r1        -1)*cos_th0                +self.costh2_CB(cos_th0)
  def costh_W12   (self, cos_th0): return  (self.r1+self.r2-1)*cos_th0
  def costh_CB1W2 (self, cos_th0): return   self.phi1*(self.costh2_W(cos_th0)+1)-1
  def costh_W1W2  (self, cos_th0): return -(self.r1        -1)*self.sinth2_W (cos_th0)+self.costh2_W (cos_th0)

  def compute_th(self, cos_th0):
    values = n.array([f(cos_th0) for f in self.funcs])
    idx = n.argmax(values)
    return (self.names[idx], values[idx])

  def plot(self, step=0.01):
    cos_th0 = n.arange(-1, 0, step)
    values = [f(cos_th0) for f in self.funcs]
    
    #plt.rc('text', usetex=True)
    plt.figure()
    
    handles = [None] * len(self.funcs)
    areIn   = n.zeros((len(self.funcs),), dtype=bool)
    
    for idx, (vals, col, name) in enumerate(zip(values, self.cols, self.names)):
      above_1 = vals>-1
      
      areIn[idx] = n.any(above_1)
      
      if areIn[idx]:
        handles[idx] = plt.plot(cos_th0[above_1], vals[above_1], col, label=name)
      
    plt.axis([-1, 0, -1, 0])
    plt.ylabel(r'cos $\theta$')
    plt.xlabel(r'cos $\theta_{0}$')
    
    plt.grid()
    plt.legend(loc='best', handles=[h[0] for h in handles if h is not None])
    plt.show()

  def trianglesUp(self, scale=1.0, dbase=-10):
    xs, zs = self.profileUp(scale)
    
    #xsAdd = n.array([xs[0], xs[-1]])#, xs[0], xs[-1]])
    #zsAdd = n.array([zs[2], zs[2]])#, dbase, dbase])
    xsAdd = n.array([xs[0], xs[-1], xs[0], xs[-1]])
    zsAdd = n.array([zs[2], zs[2], dbase, dbase])
    
    allidxs = n.arange(xs.size)
    
    idxs1 = allidxs[0::4]
    idxs2 = allidxs[1::4]
    idxs3 = allidxs[2::4]
    idxs4 = allidxs[3::4]
    
    idxsA = n.empty(idxs1.size*2)
    idxsA[0::2] = idxs1
    idxsA[1::2] = idxs2

    idxsB = n.empty(idxs1.size*2)
    idxsB[0] = xs.size
    idxsB[1:-1:2] = idxs3
    idxsB[2:-1:2] = idxs4
    idxsB[-1] = xs.size+1
    
    ts1 = n.column_stack((idxsA[0::2], idxsA[1::2], idxsB[0::2]))
    ts2 = n.column_stack((idxsA[1::2], idxsB[1::2], idxsB[0::2]))
    ts3 = n.array([[0, 1, 2], [1, 3, 2]])+xs.size
    
    return (xs, zs, xsAdd, zsAdd, n.vstack((ts1, ts2, ts3)))
    #return (xs, zs, xsAdd, zsAdd, n.vstack((ts1, ts2)))
  
  def trianglesDown(self, scale=1.0, dbase=-10):
    xs, zs = self.profileDown(scale)
    
    xsAdd = n.array([xs[0], xs[-1]])
    zsAdd = n.array([dbase, dbase])
    
    allidxs = n.arange(xs.size)
    
    idxs1 = allidxs[0::4]
    idxs1 = idxs1[1:]
    idxs2 = allidxs[1::4]
    idxs2 = idxs2[0:-1]
    idxs3 = allidxs[2::4]
    idxs4 = allidxs[3::4]
    
    ts1 = n.column_stack((idxs3, idxs4, idxs2))
    ts2 = n.column_stack((idxs4, idxs1, idxs2))
    ts3 = n.array([[0, xs.size-1, xs.size], [xs.size-1, xs.size+1, xs.size]])
    
    return (xs, zs, xsAdd, zsAdd, n.vstack((ts1, ts2, ts3)))
    #return (xs, zs, n.vstack((ts1, ts2)))
  
  def trianglesUpDown(self, scale=1.0, num=5, width=50, dbase=-10):
    #xsUp,   zsUp,   xsAddUp,   zsAddUp,   tsUp   = a.trianglesUp  (scale, dbase)
    #xsDown, zsDown, xsAddDown, zsAddDown, tsDown = a.trianglesDown(scale, dbase)
    up   = self.trianglesUp  (scale, dbase)
    down = self.trianglesDown(scale, dbase)
    xsUp,   zsUp,   xsAddUp,   zsAddUp,   tsUp   = up
    xsDown, zsDown, xsAddDown, zsAddDown, tsDown = down
    
    xsUpDown1 = n.concatenate((xsUp, xsAddUp, xsDown+xsUp[-1], xsAddDown+xsUp[-1]))
    ysUpDown1 = n.empty(xsUpDown1.shape)
    ysUpDown1.fill(0)
    zsUpDown1 = n.concatenate((zsUp, zsAddUp, zsDown, zsAddDown))
    tsUpDown1 = n.vstack((tsUp, tsDown+xsUp.size+xsAddUp.size))
    
    if num!=1:
      ysUpDown1 = n.tile(ysUpDown1, num)
      zsUpDown1 = n.tile(zsUpDown1, num)
      xx = n.empty(zsUpDown1.size)
      tt = n.empty((tsUpDown1.shape[0]*num, 3))
      nx = xsUpDown1.size
      dx = xsUp[-1]+xsDown[-1]
      ni = tsUpDown1.shape[0]
      di = xsUpDown1.size
      for i in xrange(num):
        xx[i*nx:(i+1)*nx] = xsUpDown1
        tt[i*ni:(i+1)*ni,:] = tsUpDown1
        xsUpDown1+=dx
        tsUpDown1+=di
      xsUpDown1=xx
      tsUpDown1=tt
    
    #now, make the other faces

    idxs1 = n.arange(xsUpDown1.size)
    idxs2 = n.arange(xsUpDown1.size, 2*xsUpDown1.size)
    
    idxsProfile = n.concatenate((n.ones(xsUp.shape, dtype=bool),
                                 n.zeros(xsAddUp.shape, dtype=bool),
                                 n.ones(xsDown.shape, dtype=bool),
                                 n.zeros(xsAddDown.shape, dtype=bool)))

    idxsProfile = n.tile(idxsProfile, num)
    
    idxsProfile1 = idxs1[idxsProfile]
    idxsProfile2 = idxs2[idxsProfile]
    
    tsAbove = n.vstack((n.column_stack((idxsProfile1[:-1], idxsProfile1[1:], idxsProfile2[:-1])),
                        n.column_stack((idxsProfile1[1:], idxsProfile2[1:], idxsProfile2[:-1]))))
    
    cornerA = idxs1[0]
    cornerB = idxs2[0]
    cornerC = idxs1[xsUp.size+2]
    cornerD = idxs2[xsUp.size+2]
    
    tsFront1 = n.array([[cornerA, cornerB, cornerC], [cornerB, cornerD, cornerC]])
    
    cornerA = idxs1[-3]
    cornerB = idxs2[-3]
    cornerC = idxs1[-1]
    cornerD = idxs2[-1]
    
    tsFront2 = n.array([[cornerA, cornerB, cornerC], [cornerB, cornerD, cornerC]])
    
    cornerA = idxs1[xsUp.size+2]
    cornerB = idxs2[xsUp.size+2]
    cornerC = idxs1[-1]
    cornerD = idxs2[-1]
    
    tsBottom = n.array([[cornerA, cornerB, cornerC], [cornerB, cornerD, cornerC]])
    
    xsUpDown2 = xsUpDown1
    ysUpDown2 = n.empty(ysUpDown1.shape)
    ysUpDown2.fill(width)
    zsUpDown2 = zsUpDown1
    tsUpDown2 = tsUpDown1+xsUpDown1.size
    
    tsUpDown1 = tsUpDown1[:,::-1]
    tsFront2 = tsFront2[:,::-1]
    
    xs = n.concatenate((xsUpDown1, xsUpDown2))
    ys = n.concatenate((ysUpDown1, ysUpDown2))
    zs = n.concatenate((zsUpDown1, zsUpDown2))-dbase
    ts = n.vstack((tsUpDown1, tsUpDown2, tsAbove, tsFront1, tsFront2, tsBottom))
    
    return xs, ys, zs, ts
    
  def profileUp(self, scale=1.0):
    return profile(self.k1*4-2, self.b1, self.b1+self.b2, [0, self.a2, 0, self.c2], self.k1, scale)
  
  def profileDown(self, scale=1.0):
    return profile(self.k2*4+2, self.b2, 0, [0, self.c2, 0, self.a2], self.k2+1, scale)
  
  def profileUpDown(self, scale=1.0, num=1):
    x1, y1 = self.profileUp  (scale)
    x2, y2 = self.profileDown(scale)
    x = n.concatenate((x1, x2+x1[-1]))
    y = n.concatenate((y1, y2))
    if num!=1:
      y = n.tile(y, num)
      xx = n.empty(y.shape)
      nx = x.size
      dx = x[-1]
      for i in xrange(num):
        xx[i*nx:(i+1)*nx] = x
        x+=dx
      x=xx
    #plt.plot(x, y)
    return (x,y)
  
def toSTL(xs, ys, zs, ts, filename):
  name = 'cosa'
  with open(filename, 'wt') as f:
    f.write('solid %s\n' % name)
    for t in ts:
      f.write('facet normal 0 0 0\n  outer loop\n')
      for i in t:
        f.write('    vertex %f %f %f\n' % (xs[i], ys[i], zs[i]))
      f.write('  endloop\nendfacet\n')
    f.write('endsolid %s\n' % name)

def profile(num, h1, h2, dxs, dxsnum, scale):
  """simple template for up and down profiles"""
  hs       = n.empty(num)
  hs[0::4] = h2
  hs[1::4] = h2
  hs[2::4] = h1
  hs[3::4] = h1
  dxs      = n.tile(n.array(dxs), dxsnum)
  xs       = n.cumsum(dxs[0:-2])
  return (xs*scale, hs*scale)

from cmdutils import readStringParam, readFloatParam, readIntParam

def usage():
  print "arguments: A2 B2 C2 K1 K2 B1 (stl SCALE NUM WIDTH DBASE OUTPUTFILENAME | angle THETA0)"
  print "    A2: width  of rectangular protusions at the second (small) scale"
  print "    B2: height of rectangular protusions at the second (small) scale"
  print "    C2: width  of space between rectangular protusions at the second (small) scale"
  print "    K1: number of small rectangular protusions in a big-scale rectangular protusion"
  print "    K2: number of small rectangular protusions between two big-scale rectangular protusions"
  print "    B1: height of rectangular protusions at the first (big) scale"
  print ""
  print "    MODE: either 'stl' or 'angle'"
  print ""
  print "    SCALE: scale the object"
  print "    NUM: total number number of big-scale protusions"
  print "    WIDTH: length of the object in the dimension perpendicular to the protusions"
  print "    DBASE: shift from big-scale protusions to the ground (should be negative)"
  print "    OUTPUTFILENAME: name of the STL output file"
  print ""
  print "    THETA0: angle Theta0 of the material, to compute the angle THETA for the parameters A2-B1"
  sys.exit(-1)

def main(argv):
  argidx = 1
  
  a2, argidx = readFloatParam(argv, argidx, usage, "A2")
  b2, argidx = readFloatParam(argv, argidx, usage, "B2")
  c2, argidx = readFloatParam(argv, argidx, usage, "B2")
  k1, argidx =   readIntParam(argv, argidx, usage, "K1")
  k2, argidx =   readIntParam(argv, argidx, usage, "K2")
  b1, argidx = readFloatParam(argv, argidx, usage, "B1")
  
  mode, argidx = readStringParam(argv, argidx, usage, "NO MODE!\n")
  mode    = mode.lower()
  isstl   = mode=='stl'
  isangle = mode=='angle'
  if not (isstl or isangle):
    print "ERROR: mode must be either 'stl' or 'angle'"
    usage()
  if isstl:
    sc, argidx = readFloatParam(argv, argidx, usage, "SCALE")
    nm, argidx =   readIntParam(argv, argidx, usage, "NUMBER OF BIG-SCALE PROTUSIONS")
    wd, argidx = readFloatParam(argv, argidx, usage, "WIDTH")
    db, argidx = readFloatParam(argv, argidx, usage, "BASE NEGATIVE SHIFT")
    outputfilename, argidx = readStringParam(argv, argidx, usage, "NO OUTPUT FILENAME!\n")
  else:
    angle, argidx = readFloatParam(argv, argidx, usage, "THETA0 ANGLE")

  #Theta(1.0, 1.0, 1.0, 100.0, 100.0, 200.0/3)
  #Theta(1.0, 1.0, 1.0, 10.0, 10.0, 20.0/3)
  #Theta(0.01, 0.01, 0.01, 15, 15, 0.20/3)
  theta = Theta(a2, b2, c2, k1, k2, b1, computeAllParams=not isstl)
  
  if isstl:
  
    try:
      xs, ys, zs, ts = theta.trianglesUpDown(scale=sc, num=nm, width=wd, dbase=db)
    
      toSTL(xs, ys, zs, ts, outputfilename)
    except:
      print "Unexpected exception:"
      traceback.print_exc()
      sys.exit(-1)
    
    #try:
    #  import mayavi.mlab as mlab
    #  mlab.triangular_mesh(xs, ys, zs, ts)
    #  mlab.show()
    #except:
    #  print "Unexpected exception:"
    #  traceback.print_exc()
    #  sys.exit(-1)

  else:
  
    name, value = theta.compute_th(angle)
    theta.printparams()
    print "\n Theta is defined as %s=%f\n" % (name, value)
    
if __name__=='__main__':
  main(sys.argv)
  
