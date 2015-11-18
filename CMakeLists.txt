CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

PROJECT(pyclipper)

SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")

if(NOT PYTHON_EXECUTABLE)
  if (WIN32)
    set(PYTHON_EXECUTABLE "C:/Users/jd/dev/WinPython-64bit-2.7.9.3/WinPython\ Interpreter.exe")
    if (NOT EXISTS ${PYTHON_EXECUTABLE})
      set(PYTHON_EXECUTABLE python)
    endif()
  else()
    set(PYTHON_EXECUTABLE python)
  endif()
endif()

if (CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pyclipper)
else()
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR}/pyclipper)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../clipper DESTINATION ${OUTPUTDIR}/..)

add_custom_target(MAINTARGET ALL)

add_custom_command(TARGET MAINTARGET
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUTDIR}
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/setup.py                   setup.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyclipper/__init__.py      pyclipper/__init__.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyclipper/Clipper.pyx      pyclipper/Clipper.pyx
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyclipper/Clipper.pxd      pyclipper/Clipper.pxd
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyclipper/Clipper_defs.pxd pyclipper/Clipper_defs.pxd
                   COMMAND ${PYTHON_EXECUTABLE} setup.py build_ext --inplace
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/readpaths.py               readpaths.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/contourViewer.py           contourViewer.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/slicesViewer.py            slicesViewer.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyclipper/all.py           pyclipper/all.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyclipper/plot.py          pyclipper/plot.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyclipper/plot2d.py        pyclipper/plot2d.py
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyclipper/plot3d.py        pyclipper/plot3d.py
                   WORKING_DIRECTORY ${OUTPUTDIR} VERBATIM)

