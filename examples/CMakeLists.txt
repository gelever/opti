#!/bin/sh

add_executable(linesearch linesearch.cpp)
target_link_libraries(linesearch opti)

add_executable(ndim_linesearch ndim_linesearch.cpp)
target_link_libraries(ndim_linesearch opti)

configure_file(
  "${PROJECT_SOURCE_DIR}/examples/plot_rosen.py"
  "${PROJECT_BINARY_DIR}/examples/plot_rosen.py" @ONLY)
