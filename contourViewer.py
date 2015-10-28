import pyclipper.plot2d as p2
import pyclipper.Clipper as c
import itertools as it
import sys
import time

debugfile = "c:\\users\\jd\dev\\separator\\cosa.log"

if __name__ == "__main__":
  if len(sys.argv)<3:
    sys.stderr.write('I need at least the window name and one option!\n')
    sys.exit()

  #with open(debugfile, "w") as f: f.write("hola\n")
    
  windowname = sys.argv[1]
    
  usefiles = sys.argv[2].lower() == 'files'
  #with open(debugfile, "a") as f: f.write("1\n")
  
  if usefiles:
    files = sys.argv[3:]
    paths = [None]*len(files)
  else:
    f = c.File(None, 'rb', False)
    numpaths = f.readInt64()
    paths = [None]*numpaths
    files = [f]*numpaths
  for i in xrange(len(files)):
    paths[i] = c.ClipperPaths()
    paths[i].fromStream(files[i])
  
  #linestyles = [{'linestyle':'-', 'marker':'None', 'linewidth': 2.0, 'color':p2.colorlist[i]} for i in xrange(len(files))]
  # linestyles = [{'linestyle':'None', 'marker':'o', 'markersize':5.0, 'markerfacecolor':'m'} for _ in xrange(len(files))]
  linestyles = [None]*len(files)

  defaultPatchArgs = [dict(p2.defaultPatchArgss[i % len(p2.defaultPatchArgss)]) for i in xrange(len(files))]
  #defaultPatchArgs = p2.defaultPatchArgss
  # defaultPatchArgs[2]['facecolor'] = 'none'
  # defaultPatchArgs[2]['edgecolor'] = '#100000'
  # defaultPatchArgs[2]['lw'] = 2
  # defaultPatchArgs[3]['facecolor'] = 'none'
  # defaultPatchArgs[3]['edgecolor'] = '#001000'
  # defaultPatchArgs[3]['lw'] = 2
  
  for i in xrange(len(files)):
    defaultPatchArgs[i]['facecolor'] = 'none'
    defaultPatchArgs[i]['edgecolor'] = p2.colorlist[i % len(p2.colorlist)]
    defaultPatchArgs[i]['linewidth'] = 1

  #for toolpaths which are not contours
  nocontours=[]#[4]
  for i in nocontours:
    defaultPatchArgs[i]['edgecolor'] = 'None'
    # linestyles[i] = {'linestyle':'-', 'marker':'None', 'linewidth': 2.0, 'color':p2.colorlist[i]}
    linestyles[i] = {'facecolors': 'None', 'edgecolors': p2.colorlist[i], 'linewidths': 2.0}
    
  p2.showSlices(paths, modeN=True, title=windowname, linestyle=linestyles, patchArgs=defaultPatchArgs)#,BB=bb, fig=makeMaximizedFig())
  
    
