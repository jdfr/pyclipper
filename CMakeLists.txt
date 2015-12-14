CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

option(GENERATE_PYTHON_CALLERS "generate command line scripts to call python scripts" OFF)

PROJECT(pyclipper)

SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")

set(SRC ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(BASE_OUTPUTDIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
else()
  set(BASE_OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(OUTPUTDIR ${BASE_OUTPUTDIR}/pyclipper)

#if present, include code from this file. Useful for setting some configuration values such as PYTHON_EXECUTABLE and PYTHON_EXECUTABLE_FOR_SETUP
#Why using separate PYTHON_EXECUTABLE and PYTHON_EXECUTABLE_FOR_SETUP? Because WinPython, the python distribution we are using in windows, requires to launch a specially-configured interpreter if you are building extensions, otherwise the build process fails
include(${SRC}/cmake.configvars.txt OPTIONAL)

if (PYTHON_EXECUTABLE_FOR_SETUP)
  if (IS_ABSOLUTE ${PYTHON_EXECUTABLE_FOR_SETUP})
    if (NOT EXISTS ${PYTHON_EXECUTABLE_FOR_SETUP})
      unset(PYTHON_EXECUTABLE_FOR_SETUP)
    endif()
  endif()
endif()

if(NOT PYTHON_EXECUTABLE_FOR_SETUP)
  if (WIN32)
    message(WARNING "PYTHON_EXECUTABLE_FOR_SETUP variable is not set. Setting it to \"python\". If you are using a non-installed WinPython distribution, it MUST be the path to \"WinPython Interpreter.exe\", otherwise errors will follow.")
  endif()
  set(PYTHON_EXECUTABLE_FOR_SETUP python CACHE PATH "python executable for use with setup.py (if using WinPython, must be \"WinPython Interpreter.exe\"")
endif()

file(COPY ${SRC}/../clipper DESTINATION ${OUTPUTDIR}/..)

add_custom_target(pyclipper ALL)

#up to the invocation of PYTHON_EXECUTABLE_FOR_SETUP, the order is important!!!!
add_custom_command(TARGET pyclipper
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUTDIR}
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/setup.py                   setup.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/__init__.py      pyclipper/__init__.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper.pyx      pyclipper/Clipper.pyx
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper.pxd      pyclipper/Clipper.pxd
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/Clipper_defs.pxd pyclipper/Clipper_defs.pxd
                   COMMAND ${PYTHON_EXECUTABLE_FOR_SETUP} setup.py build_ext --inplace
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/iopaths.py                 iopaths.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/contourViewer.py           contourViewer.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/slicesViewer.py            slicesViewer.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/all.py           pyclipper/all.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot.py          pyclipper/plot.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot2d.py        pyclipper/plot2d.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC}/pyclipper/plot3d.py        pyclipper/plot3d.py
                   WORKING_DIRECTORY ${OUTPUTDIR} VERBATIM)

