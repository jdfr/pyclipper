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

#from ._pyslic3r import *

#VERY UGLY HACK TO GET LD TO LINK TO './libslic3rlib.so' WITHOUT INSTALLING IT 
def _dynamicLoadHack():
  from ctypes import cdll
  import os.path
  cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libslic3rlib.so'))

try:
  from ._pyslic3r import *
except ImportError:
  #probably the linker cannot find libslicerlib.so because it is not installed.
  #Trying a hack...
  _dynamicLoadHack()
  #trying again...
  from ._pyslic3r import *

