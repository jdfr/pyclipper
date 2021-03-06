CMAKE_MINIMUM_REQUIRED(VERSION 3.0) #maybe it actually requires higher, but at most 3.5...

include(config.cmake)

PROJECT(pyclipper LANGUAGES) #no compiled language, skip usual compiler  tests

SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")

set(SRC ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(BASE_OUTPUTDIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
else()
  set(BASE_OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(OUTPUTDIR ${BASE_OUTPUTDIR}/pyclipper)

if( ("${PYTHON_EXECUTABLE}" STREQUAL "") OR ("${PYTHON_EXECUTABLE_FOR_SETUP}" STREQUAL "") )
  find_program(PYTHON_EXECUTABLE_FOUND python)
  if(PYTHON_EXECUTABLE_FOUND)
    if("${PYTHON_EXECUTABLE}" STREQUAL "")
      set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE_FOUND}")
    endif()
    if("${PYTHON_EXECUTABLE_FOR_SETUP}" STREQUAL "")
      set(PYTHON_EXECUTABLE_FOR_SETUP "${PYTHON_EXECUTABLE_FOUND}")
    endif()
  else()
    message(FATAL_ERROR "Please set up the path to python in PYTHON_EXECUTABLE and PYTHON_EXECUTABLE_FOR_SETUP (the latter one might be different in some systems, such as WinPython)")
  endif()
endif()

if (NOT FILESEP)
  if (WIN32)
    set(FILESEP \\)
  else()
    set(FILESEP /)
  endif()
endif()
string(REPLACE / ${FILESEP} PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
string(REPLACE / ${FILESEP} PYTHON_EXECUTABLE_FOR_SETUP "${PYTHON_EXECUTABLE_FOR_SETUP}")

if(GENERATE_PYTHON_CALLERS)

  function(GENERATE_SCRIPT SCRIPT_NAME)
    if (WIN32)
      file(WRITE ${BASE_OUTPUTDIR}/${SCRIPT_NAME}.bat "\"${PYTHON_EXECUTABLE}\" pyclipper\\${SCRIPT_NAME}.py %*")
    else()
      file(WRITE ${OUTPUTDIR}/${SCRIPT_NAME} "#!/bin/sh\n${PYTHON_EXECUTABLE} pyclipper/${SCRIPT_NAME}.py \"$@\"")
      file(COPY ${OUTPUTDIR}/${SCRIPT_NAME} DESTINATION ${BASE_OUTPUTDIR} FILE_PERMISSIONS
           OWNER_EXECUTE OWNER_READ OWNER_WRITE
           GROUP_EXECUTE GROUP_READ GROUP_WRITE
           WORLD_EXECUTE WORLD_READ)
      file(REMOVE ${OUTPUTDIR}/${SCRIPT_NAME})
    endif()
  endfunction()
  
  GENERATE_SCRIPT(slicesViewer)
  GENERATE_SCRIPT(cloudmesh)
  GENERATE_SCRIPT(twoscales)
  if (WIN32)
    file(WRITE ${BASE_OUTPUTDIR}/py.bat           "\"${PYTHON_EXECUTABLE}\" %*")
  endif()
endif()

if("${CLIPPER_BASE_DIR}" STREQUAL "")
  if (EXISTS ${SRC}/../clipper)
    set(CLIPPER_BASE_DIR ${SRC}/../clipper)
  else()
    message(FATAL_ERROR "Please set the variable CLIPPER_BASE_DIR to the base path to the clipper/iopaths directory")
  endif()
endif()

file(COPY ${CLIPPER_BASE_DIR} DESTINATION ${OUTPUTDIR})

#up to the invocation of PYTHON_EXECUTABLE_FOR_SETUP, the order is important!!!!
add_custom_target(pyclipper ALL
                  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUTDIR}
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/setup.py                   setup.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/__init__.py      pyclipper/__init__.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper.pyx      pyclipper/Clipper.pyx
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper.pxd      pyclipper/Clipper.pxd
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper_defs.pxd pyclipper/Clipper_defs.pxd
                  COMMAND ${PYTHON_EXECUTABLE_FOR_SETUP} setup.py build_ext --inplace
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/stl.py                     stl.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/mergestls.py               mergestls.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/cmdutils.py                cmdutils.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/cloudmesh.py               cloudmesh.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/iopaths.py                 iopaths.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/contourViewer.py           contourViewer.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/slicesViewer.py            slicesViewer.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/all.py           pyclipper/all.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/minisix.py       pyclipper/minisix.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot.py          pyclipper/plot.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot2d.py        pyclipper/plot2d.py
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot3d.py        pyclipper/plot3d.py
                  WORKING_DIRECTORY ${OUTPUTDIR} VERBATIM)

