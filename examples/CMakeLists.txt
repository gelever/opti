#!/bin/sh

add_executable(linesearch linesearch.cpp)
target_link_libraries(linesearch opti)

add_executable(trustregion trustregion.cpp)
target_link_libraries(trustregion opti)

add_executable(partest partest.cpp)
target_link_libraries(partest opti)

configure_file(
  "${PROJECT_SOURCE_DIR}/examples/plot_rosen.py"
  "${PROJECT_BINARY_DIR}/examples/plot_rosen.py" @ONLY)
