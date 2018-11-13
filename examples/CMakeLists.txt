#!/bin/sh

add_executable(linesearch linesearch.cpp)
target_link_libraries(linesearch opti)

add_executable(trustregion trustregion.cpp)
target_link_libraries(trustregion opti)

add_executable(nonlinear nonlinear.cpp)
target_link_libraries(nonlinear opti)

configure_file(
  "${PROJECT_SOURCE_DIR}/examples/plot_rosen.py"
  "${PROJECT_BINARY_DIR}/examples/plot_rosen.py" @ONLY)
