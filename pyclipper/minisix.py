import sys

PY3 = sys.version_info >= (3,0)

if PY3:
  xrange = range
  izip   = zip
else:
  from itertools import izip