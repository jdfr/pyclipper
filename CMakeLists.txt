CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

PROJECT(pyclipper)

SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")

set(SRC ${CMAKE_CURRENT_SOURCE_DIR})

#if present, include code from this file. Useful for setting some configuration values such as PYTHON_EXECUTABLE
include(${SRC}/cmake.configvars.txt OPTIONAL)

if (PYTHON_EXECUTABLE)
  if (IS_ABSOLUTE ${PYTHON_EXECUTABLE})
    if (NOT EXISTS ${PYTHON_EXECUTABLE})
      unset(PYTHON_EXECUTABLE)
    endif()
  endif()
endif()
if(NOT PYTHON_EXECUTABLE)
  if (WIN32)
    message(WARNING "PYTHON_EXECUTABLE variable is not set. Setting it to \"python\". If you are using a non-installed WinPython distribution, it MUST be the path to \"WinPython Interpreter.exe\", otherwise errors will follow.")
  endif()
  set(PYTHON_EXECUTABLE python CACHE PATH "python executable (if using WinPython, must be \"WinPython Interpreter.exe\"")
endif()

if (CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pyclipper)
else()
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR}/pyclipper)
endif()

file(COPY ${SRC}/../clipper DESTINATION ${OUTPUTDIR}/..)

add_custom_target(MAINTARGET ALL)

#up to the invocation of PYTHON_EXECUTABLE, the order is important!!!!
add_custom_command(TARGET MAINTARGET
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUTDIR}
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/setup.py                   setup.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/__init__.py      pyclipper/__init__.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper.pyx      pyclipper/Clipper.pyx
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper.pxd      pyclipper/Clipper.pxd
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper_defs.pxd pyclipper/Clipper_defs.pxd
                   COMMAND ${PYTHON_EXECUTABLE} setup.py build_ext --inplace
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/readpaths.py               readpaths.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/contourViewer.py           contourViewer.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/slicesViewer.py            slicesViewer.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/all.py           pyclipper/all.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot.py          pyclipper/plot.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot2d.py        pyclipper/plot2d.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot3d.py        pyclipper/plot3d.py
                   WORKING_DIRECTORY ${OUTPUTDIR} VERBATIM)

