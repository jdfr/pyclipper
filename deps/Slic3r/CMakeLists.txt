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

#to build our dependency on Slic3r's C++ library, change to directory Slic3r-build and do
#    cmake ..
#if cmake does not complain, you can then build the project with
#    make

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0)

PROJECT(slic3rlib)


SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")

ADD_DEFINITIONS(-fPIC)

#set (DODEBUG "-DSLIC3R_DEBUG")
set (DODEBUG "")

#add compiler flags from Slicer/xs/Build.PL (HAS_BOOL and SLIC3RXS
#are specific to the perl build, so they are not included here)
# _GLIBCXX_USE_C99 : to get the long long type for g++
# HAS_BOOL         : stops Perl/lib/CORE/handy.h from doing "#  define bool char" for MSVC
# NOGDI            : prevents inclusion of wingdi.h which defines functions Polygon() and Polyline() in global namespace
set (SLIC3RFLAGS "-D_GLIBCXX_USE_C99 -DNOGDI ${DODEBUG}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC ${SLIC3RFLAGS}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

set(SRC Slic3r/xs/src)


#*.c, *.cc, *.cpp
file(GLOB_RECURSE c_sources "${SRC}/*.c*")
file(GLOB_RECURSE h_sources "${SRC}/*.h")
#add silly config.h dependency in one of Slic3r's subprojects
set(CONFIGH Slic3r-include)

set (slic3r_INCLUDE_DIRS ${CONFIGH} ${SRC} ${SRC}/admesh ${SRC}/boost ${SRC}/libslic3r ${SRC}/poly2tri)
set(SOURCES ${c_sources} ${h_sources})
include_directories(${slic3r_INCLUDE_DIRS})

## to see the include directories
#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#  message(STATUS "dir='${dir}'")
#endforeach()

ADD_LIBRARY(slic3rlib SHARED ${c_sources})
#ADD_LIBRARY(slic3rlib STATIC ${c_sources})

