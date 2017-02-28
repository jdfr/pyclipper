from __future__ import print_function

def readStringParam(argv, argidx, usage, message):
  if len(argv)<=argidx:
    print(message)
    usage()
  return (argv[argidx], argidx+1)

def readFloatParam(argv, argidx, usage, paramName):
  if len(argv)<=argidx:
    print("NO %s!\n" % paramName)
    usage()
  try:
    param = float(argv[argidx])
  except:
    print("%s (%s) IS NOT A FLOATING POINT NUMBER!\n" % (paramName, argv[argidx]))
    usage()
  return (param, argidx+1)

def readIntParam(argv, argidx, usage, paramName):
  if len(argv)<=argidx:
    print("NO %s!\n" % paramName)
    usage()
  try:
    param = int(argv[argidx])
  except:
    print("%s (%s) IS NOT AN INTEGER NUMBER!\n" % (paramName, argv[argidx]))
    usage()
  return (param, argidx+1)
