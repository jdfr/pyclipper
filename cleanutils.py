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

import os
import os.path as op
import shutil

def erasedircontents(basedir):
  for name in os.listdir(basedir):
    absname = op.join(basedir, name)
    if op.isfile(absname):
      os.remove(absname)
    else:
      shutil.rmtree(absname)

# clean previous build
def doClean(dirname, pynames):
  
  #remove ../build
  superbuild = op.join(dirname, '../build')
  if op.isdir(superbuild):
    shutil.rmtree(superbuild)
    
  basedir = op.abspath(op.join(dirname, '..'))
  #remove ../*.pyc
  for name in os.listdir(basedir):
    if name.endswith(".pyc"):
      os.remove(op.join(basedir, name))
  
  for name in os.listdir(dirname):
    absname  = op.join(dirname, name)
    if op.isfile(absname):
      toRemove = ( name.endswith(".pyc") or  #remove *.pyc
                   any(   name.startswith(pyname) and #remove ./*.pyc and ./{CYTHONMODULE}.{so, cpp}
                          not any(name.endswith(x) for x in [".pyx", ".pxd", ".pxi"])
                        for pyname in pynames) )
      if toRemove:
        print "Removing file "+absname
        os.remove(absname)
    #remove ./build
    elif op.isdir(absname) and (name == "build"):
      shutil.rmtree(absname)
