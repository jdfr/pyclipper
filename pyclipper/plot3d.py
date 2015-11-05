# Copyright (c) 2015 Jose David Fernandez Rodriguez
#  
# This file is distributed under the terms of the
# GNU Affero General Public License, version 3
# as published by the Free Software Foundation.
# 
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public
# License along with this file. You may obtain a copy of the License at
# http://www.gnu.org/licenses/agpl-3.0.txt

#VALID MAYAVI COLORMAPS:
#accent       flag          hot      pubu     set2
#autumn       gist_earth    hsv      pubugn   set3
#black-white  gist_gray     jet      puor     spectral
#blue-red     gist_heat     oranges  purd     spring
#blues        gist_ncar     orrd     purples  summer
#bone         gist_rainbow  paired   rdbu     winter
#brbg         gist_stern    pastel1  rdgy     ylgnbu
#bugn         gist_yarg     pastel2  rdpu     ylgn
#bupu         gnbu          pink     rdylbu   ylorbr
#cool         gray          piyg     rdylgn   ylorrd
#copper       greens        prgn     reds
#dark2        greys         prism    set1

import itertools as it

import numpy     as n

from mayavi import  mlab

def showSlices(planar_paths_list, title=None, modes=None, argss=None):
  if modes is None:
    modes = ['line']
  if argss is None:
    argss = [{}]
  figargs = {}
  if title:
    figargs['figure'] = title
  mlab.figure(**figargs)
  for (planar_paths, mode, args) in it.izip(planar_paths_list, it.cycle(modes), it.cycle(argss)):
    showSlicesType(planar_paths, mode, args)
  mlab.show()

def showSlicesType(planar_paths, mode=None, args={}):
  if mode=='tube':
    for z, paths in planar_paths:
      for path in paths:
        zv = n.empty((path.shape[0],))
        zv.fill(z)
        mlab.plot3d(path[:,0], path[:,1], zv, **args)
  else:
    uselines = mode=='line'
    #make a list of pairs (cycle, z), composed from both contours and holes with their respective z's
    allcycles = list(it.chain.from_iterable( zip(paths, it.cycle((z,)))
                                             for z,paths in planar_paths))
    #get cycle sizes    
    cyclessizes = list(cycle.shape[0] for cycle, z in allcycles)
    #get cumulative starting index for each cycle
    cyclestartidxs = n.roll(n.cumsum(cyclessizes), 1)
    cyclestartidxs[0] = 0
    #concatenate XY coords for all cycles
    #cyclesxy = n.vstack([cycle for cycle,_ in allcycles])
    cyclesx  = n.empty((sum(cyclessizes),))
    cyclesy  = n.empty((cyclesx.shape[0],))
    #size matrices for (a) concatenated z values and (b) line connections for all cycles
    cyclesz  = n.empty((cyclesx.shape[0],))
    conns  = n.empty((cyclesx.shape[0],2))
    #iterate over each cycle's starting index, size, and z
    for startidx, size, (cycle,z) in it.izip(cyclestartidxs, cyclessizes, allcycles):
      endidx = startidx+size
      cyclesx[startidx:endidx] = cycle[:,0]       #set x for the current cycle
      cyclesy[startidx:endidx] = cycle[:,1]       #set y for the current cycle
      cyclesz[startidx:endidx] = z                #set z for the current cycle
      rang = n.arange(startidx, endidx)
      conns[startidx:endidx,0] = rang    #set line connections for the current cycle
      conns[startidx+1:endidx,1] = rang[:-1]
      conns[startidx, 1] = rang[-1]
      if uselines:
        conns[startidx, 1] = rang[1]
    #put all the processed data into mayavi
    #cyclesx *= scalingFactor
    #cyclesy *= scalingFactor
    src = mlab.pipeline.scalar_scatter(cyclesx,cyclesy,cyclesz)
    src.mlab_source.dataset.lines = conns # Connect them
    lines = mlab.pipeline.stripper(src) # The stripper filter cleans up connected lines
    mlab.pipeline.surface(lines, **args)#, line_width=1)#, opacity=.4) # Finally, display the set of lines
      