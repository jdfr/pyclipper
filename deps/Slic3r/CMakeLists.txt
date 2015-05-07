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

#this file is called from the main CMakeLists.txt in the root directory, but can also be called
#standalone from the directory deps/Slic3r/cmakebuild. In that directory, do
#    cmake ..
#then, if cmake does not complain, you can then build the project with
#    make

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/Slic3r/xs/src)

#check if the Slic3r source is present, download it otherwise
if (NOT EXISTS ${SRC})
  #we now execute a git command. cmake has a directive to build external projects (ExternalProject_Add)
  #that can do this, but it does so at building time, while we need to download the source at cmake time
  execute_process(COMMAND git clone https://github.com/jdfr/Slic3r.git
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE gitresult)
  if (NOT (gitresult EQUAL 0))
    message(FATAL_ERROR "No source directory for Slic3r was detected (deps/Slic3r/Slic3r/xs/src/), but git was not able to clone Slic3r into deps/Slic3r/Slic3r")
  endif()
endif()


PROJECT(slic3rlib)

SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")

ADD_DEFINITIONS(-fPIC)

#add compiler flags from Slicer/xs/Build.PL (HAS_BOOL and SLIC3RXS
#are specific to the perl build, so they are not included here)
# _GLIBCXX_USE_C99 : to get the long long type for g++
# NOGDI            : prevents inclusion of wingdi.h which defines functions Polygon() and Polyline() in global namespace
ADD_DEFINITIONS(-D_GLIBCXX_USE_C99 -DNOGDI)

#USE THESE OPTIONS FROM THE COMMAND LINE IN THIS WAY:
#          cmake .. -DDODEBUG=ON
#          cmake .. -DDOTIMING=ON
#BUT BE AWARE THAT THEY WILL IMPLY RECOMPILATION OF ALL FILES
option(DODEBUG           "compile with debug  messages" OFF)
option(DOTIMING          "compile with timing messages" OFF)
option(DONOTUSEPOLY2TRI  "disable functions that require the use of poly2tri" ON)
option(FULLLIBRARY       "include all sources in the library" OFF)

set(MASK "/*.c*") #*.c, *.cc, *.cpp
file(GLOB_RECURSE poly2tri_c  "${SRC}/poly2tri${MASK}") 
file(GLOB_RECURSE admesh_c    "${SRC}/admesh${MASK}")
set(clipper_source "${SRC}/clipper.cpp") #clipper goes into its own library
file(GLOB         slic3r1_c   "${SRC}${MASK}")
list(REMOVE_ITEM  slic3r1_c   ${clipper_source})
if (FULLLIBRARY)
  file(GLOB_RECURSE slic3r2_c "${SRC}/libslic3r${MASK}")
else()
  #include just the sources we need right now
  set (selectedfiles BoundingBox ClipperUtils ExPolygon ExPolygonCollection Geometry Line MultiPoint Point Polygon Polyline TriangleMesh utils)
  set(slic3r2_c "")
  foreach(f ${selectedfiles})
    list(APPEND slic3r2_c ${SRC}/libslic3r/${f}.cpp)
  endforeach()
endif()

set(slic3r_sources ${admesh_c} ${slic3r1_c} ${slic3r2_c})

if (FULLLIBRARY OR NOT DONOTUSEPOLY2TRI)
  set(slic3r_sources ${slic3r_sources} ${poly2tri_c})
endif()
set(SOURCES ${slic3r_sources} )

set(CONFIGH externalincludes) #add silly config.h dependency in one of Slic3r's subprojects
set (slic3r_INCLUDE_DIRS ${CONFIGH} ${SRC} ${SRC}/admesh ${SRC}/boost ${SRC}/libslic3r ${SRC}/poly2tri)
include_directories(${slic3r_INCLUDE_DIRS})

if (DODEBUG)
  ADD_DEFINITIONS(-DSLIC3R_DEBUG)
endif()
if (DOTIMING)
  set_source_files_properties (${SRC}/libslic3r/TriangleMesh.cpp ${SRC}/admesh/normals.c ${SRC}/admesh/stlinit.c COMPILE_FLAGS -DPRINT_TIMING)
endif()
if (DONOTUSEPOLY2TRI)
  ADD_DEFINITIONS(-DNOPOLYT2RI)
endif()

#foreach(file ${slic3r1_c})
#  message(STATUS "file='${file}'")
#endforeach()
## to see the include directories
#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#  message(STATUS "dir='${dir}'")
#endforeach()

ADD_LIBRARY(clipper SHARED ${clipper_source})
ADD_LIBRARY(slic3rlib SHARED ${slic3r_sources})
target_link_libraries(slic3rlib clipper)
#ADD_LIBRARY(slic3rlib STATIC ${slic3r_sources})

get_filename_component(BASEPATH ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set (CMAKE_INSTALL_PREFIX "${BASEPATH}/../..")
install(TARGETS slic3rlib DESTINATION "pyslic3r")