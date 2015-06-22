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

#This is the main CMakeLists.txt. It builds libraries from Clipper and Slic3r's C++ sources,
#then builds the cython extensions for the python bndings of these libraries.
#To compile this project, do the following from the command line:
#    cd cmakebuild
#    cmake ..
#    make

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0)

set(slic3rlib ${CMAKE_CURRENT_SOURCE_DIR}/deps/Slic3r)


add_subdirectory (${slic3rlib} ${slic3rlib}/cmakebuild)

PROJECT(pyslic3r)

SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")

set(pydir ${CMAKE_CURRENT_SOURCE_DIR}/pyslic3r)

set(inputs clipper slic3rlib)

set(sourcefiles SlicedModel TriangleMesh Clipper)
set(outputs "")
foreach(f ${sourcefiles})
  list(APPEND outputs ${pydir}/${f}.so)
  list(APPEND  inputs ${pydir}/${f}.pyx)
endforeach()

add_custom_target(pyslic3r ALL
                  COMMAND python setup.py build_ext --inplace
                  DEPENDS ${inputs}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Building pyslic3r's cython extensions..."
                  VERBATIM)

add_custom_target(pyclean
                  COMMAND python clearcython.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Removing pyslic3r's cython extensions..."
                  VERBATIM)

add_custom_target(cmakeclean
                  COMMAND python clearcmake.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Removing cmake directories..."
                  VERBATIM)
