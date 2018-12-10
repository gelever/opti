#!/bin/sh

add_executable(linesearch linesearch.cpp)
target_link_libraries(linesearch opti)

add_executable(trustregion trustregion.cpp)
target_link_libraries(trustregion opti)

add_executable(nonlinear nonlinear.cpp)
target_link_libraries(nonlinear opti)

function(copy_file filename)
    configure_file(
        "${PROJECT_SOURCE_DIR}/examples/${filename}"
        "${PROJECT_BINARY_DIR}/examples/${filename}" @ONLY)
endfunction(copy_file)

copy_file(plot_rosen.py)
copy_file(tikhonov.py)
copy_file(dollarblur.m)
