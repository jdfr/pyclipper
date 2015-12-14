import sys
import readpaths as rp

debugfile = "cosa.log"


scalingFactor          = 0.00000001

craw = '#cccccc' #gray
raw_fac = 0.7
craw3d = (raw_fac, raw_fac, raw_fac)

cmap_raw = 'Greys'

cmaps_toolpaths = [
         'Reds',
         'Blues',
         'Greens',
         ]

cmaps_contours = [
         'Oranges',
         'Purples',
         'spring',
         ]

ctoolpaths = [
             '#ff0000', #red
             '#0000ff', #blue
             '#00ff00', #green
             '#00ffff', #cyan
             '#ffff00', #yellow
             '#ff00ff', #magenta
              ]

ccontours = [col.replace('ff', '44') for col in ctoolpaths]

ctoolpaths3d = [
             (1,0,0), #red
             (0,1,0), #blue
             (0,0,1), #green
             (0,1,1), #cyan
             (1,1,0), #yellow
             (1,0,1), #magenta
              ]
contour_fac = 0.4
ccontours3d = [tuple(c*contour_fac for c in col) for col in ctoolpaths3d]

ncols = len(ctoolpaths)
ncmaps = len(cmaps_toolpaths)

#index in the list (to be passed to showSlices) of each path type-ntool
def showlistidx(typ, ntool):
  return 0 if typ==rp.TYPE_RAW_CONTOUR else contents.numtools*(0 if typ==rp.TYPE_PROCESSED_CONTOUR else 1)+ntool+1

def type2str(typ):
  if   typ==rp.TYPE_RAW_CONTOUR:
    return 'raw'
  elif typ==rp.TYPE_PROCESSED_CONTOUR:
    return 'contour'
  elif typ==rp.TYPE_TOOLPATH:
    return 'toolpath'

def show2D(contents, windowname, custom_formatting):
  nocustom  = custom_formatting is None
  #generate an ordered list of lists of paths according to showlistidx.
  #The lists of paths are sorted by z, with empty places if needed,
  #so all lists have the same number of elements, and are Z-ordered
  nelems            = contents.numtools*2+1
  pathsbytype_list  = [None]*nelems
  scalings_list     = [None]*nelems
  usePatches_list   = [None]*nelems
  linestyles_list   = [None]*nelems
  patchestyles_list = [None]*nelems
  for key in contents.records.keys():
    typ, ntool = key
    if not typ in rp.ALL_TYPES:
      raise Exception('Unrecognized path type %d' % typ)
    byz   = contents.records[key]
    byzl  = []
    byzls = []
    for z in contents.zs:
      if z in byz:
        if byz[z].savemode==rp.SAVEMODE_DOUBLE_3D:
          raise Exception('path with type=%d, ntool=%d, z=%f is 3D, cannot show it in matplotlib!!!!' % (typ, ntool, z))
        byzl .append(byz[z].paths)
        byzls.append(byz[z].scaling)
      else:
        byzl .append([])
        byzls.append([])
    idx                      = showlistidx(typ, ntool)
    pathsbytype_list[idx]    = byzl
    scalings_list   [idx]    = byzls
    if nocustom:
      if   typ==rp.TYPE_RAW_CONTOUR:
        usePatches_list[idx]   = True
        linestyles_list[idx]   = None
        patchestyles_list[idx] = {'facecolor':craw, 'edgecolor':'none', 'lw': 1}
      elif typ==rp.TYPE_PROCESSED_CONTOUR:
        usePatches_list[idx]   = True
        linestyles_list[idx]   = None
        patchestyles_list[idx] = {'facecolor':ccontours[ntool%ncols], 'edgecolor':'none', 'lw': 1}
      elif typ==rp.TYPE_TOOLPATH:
        usePatches_list[idx]   = False
        linestyles_list[idx]   = {'linewidths':2, 'colors': ctoolpaths[ntool%ncols]}
        patchestyles_list[idx] = None
    else:
      typs = type2str(typ)
      if typ==rp.TYPE_RAW_CONTOUR:
        usePatches_list[idx]   = custom_formatting[typs]['usepatches']
        linestyles_list[idx]   = custom_formatting[typs]['linestyle']
        patchestyles_list[idx] = custom_formatting[typs]['patchstyle']
      else:
        length                 = len(custom_formatting[typs])
        usePatches_list[idx]   = custom_formatting[typs][ntool%length]['usepatches']
        linestyles_list[idx]   = custom_formatting[typs][ntool%length]['linestyle']
        patchestyles_list[idx] = custom_formatting[typs][ntool%length]['patchstyle']

  p2.showSlices(pathsbytype_list, modeN=True, title=windowname, BB=[], zs=contents.zs, linestyle=linestyles_list, patchArgs=patchestyles_list, usePatches=usePatches_list, scalingFactor=scalings_list)

def show3D(contents, windowname, custom_formatting):
  nocustom  = custom_formatting is None
  nelems          = contents.numtools*2+1
  paths_list      = [None]*nelems
  mode_list       = [None]*nelems
  args_list       = [None]*nelems
  for key in contents.records.keys():
    typ, ntool = key
    if not typ in rp.ALL_TYPES:
      raise Exception('Unrecognized path type %d' % typ)
    byz = contents.records[key]
    byzl = []
    for z in contents.zs:
      if z in byz:
        byzl.append([z, byz[z].paths, byz[z].scaling])
    idx                      = showlistidx(typ, ntool)
    paths_list[idx]          = byzl
    if nocustom:
      if   typ==rp.TYPE_RAW_CONTOUR:
        mode_list[idx]         = 'contour'
        args_list[idx]         = {'color':    craw3d, 'line_width':2}
        #args_list[idx]         = {'colormap':cmap_raw, 'line_width':2}
      elif typ==rp.TYPE_PROCESSED_CONTOUR:
        mode_list[idx]         = 'line'
        args_list[idx]         = {'color':      ccontours3d[ntool%ncols],  'line_width':2}
        #args_list[idx]         = {'colormap':cmaps_contours[ntool%ncmaps], 'line_width':2}
      elif typ==rp.TYPE_TOOLPATH:
        mode_list[idx]         = 'line'
        args_list[idx]         = {'color':      ctoolpaths3d[ntool%ncols],  'line_width':2}
        #TODO: decimate the lines (maybe implement a Douglas-Peucker?) before adding the tubes!!!!!
        #mode_list[idx]         = 'tube'
        #args_list[idx]         = {'color':      ctoolpaths3d[ntool%ncols],  'line_width':2, 'tube_radius':contents.xradiuses[ntool]}
        ##args_list[idx]         = {'colormap':cmaps_toolpaths[ntool%ncmaps], 'line_width':2}
    else:
      typs = type2str(typ)
      if typ==rp.TYPE_RAW_CONTOUR:
        mode_list[idx]         = custom_formatting[typs]['mode']
        args_list[idx]         = custom_formatting[typs]['args']
      else:
        length                 = len(custom_formatting[typs])
        mode_list[idx]         = custom_formatting[typs][ntool%length]['mode']
        args_list[idx]         = custom_formatting[typs][ntool%length]['args']
        if mode_list[idx]=='tube':
          args_list[idx]['tube_radius'] = contents.xradiuses[ntool]

  p3.showSlices(paths_list, title=windowname, modes=mode_list, argss=args_list)
  
def check_args(cond, errmsg):
  if cond:
    USAGE = ("\n\nUSAGE: %s WINDOWNAME (2d|3d) (pipe | file INPUTFILE) [CUSTOMFORMATTING]\n"
             "    WINDOWNAME: name of the window\n"
             "    2d/3d: will display in 2D (matplotlib) or 3D (mayavi) mode\n"
             "    pipe: will read data from binary stdin\n"
             "    file INPUTFILENAME: will read data from input file\n"
             "    CUSTOMFORMATTING: if present, it is evaluated to a python structure containing formatting info. It is 2d/3d mode dependent, and no check is done, so it is very brittle, please see the code."
             ) % sys.argv[0]
    sys.stderr.write(errmsg+USAGE)
    sys.exit()

if __name__ == "__main__":
  argidx=1
  argc = len(sys.argv)
  
  check_args(argidx>=argc, 'need to read the window name (first argument), but no more arguments!')
  windowname = sys.argv[argidx]
  argidx+=1
  
  check_args(argidx>=argc, 'need to read the display mode (second argument), but no more arguments!')
  use2d = sys.argv[argidx].lower()
  check_args(not use2d in ['2d', '3d'], 'second argument must be either "2d" or "3d", but it is %s!!!' % use2d)
  use2d = use2d=='2d'
  argidx+=1
  
  check_args(argidx>=argc, 'need to read the input mode argument (third argument), but no more arguments!')
  usefile = sys.argv[argidx].lower()
  check_args(not usefile in ['file', 'pipe'], 'third argument must be either "file" or "pipe", but it is %s!!!' % usefile)
  usefile = usefile == 'file'
  argidx+=1
  
  if usefile:
    check_args(argidx>=argc, 'need to read the file name, but no more arguments!')
    filename = sys.argv[argidx]
    argidx+=1
  else:
    filename = None
  
  if argidx<argc:
    try:
      custom_formatting = eval(sys.argv[argidx], None, None)
    except:
      check_args(True, 'Could not evaluate the parameter for custom formatting!!!')
    argidx+=1
  else:
    custom_formatting = None

  contents = rp.FileContents()
  contents.readFromFile(filename)
  contents.organizeRecords()
  
  if use2d:
    import pyclipper.plot2d as p2
    show2D(contents, windowname, custom_formatting)
  else:
    import pyclipper.plot3d as p3
    show3D(contents, windowname, custom_formatting)
  
