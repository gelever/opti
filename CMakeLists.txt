#!/bin/sh

# Copyright (c) 2018, Stephan Gelever

cmake_minimum_required(VERSION 3.5)
project(opti VERSION 1.0 LANGUAGES CXX)
enable_testing()

find_package(linalgcpp REQUIRED)

add_library(opti
    src/utilities.cpp
    src/opti.cpp
)

target_compile_options(opti PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -O2>)
target_compile_options(opti PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -O2>)

target_link_libraries(opti
    PUBLIC
        linalgcpp::linalgcpp
        linalgcpp::parlinalgcpp
)

target_include_directories(opti
    PUBLIC
		$<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
)

# Installation
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/libopti)

install(TARGETS opti
    EXPORT opti-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets to a script
install(EXPORT opti-targets
    FILE optiTargets.cmake
    NAMESPACE opti::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/optiConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/optiConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/optiConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

configure_file(
    "${PROJECT_SOURCE_DIR}/config/opti_config.h.in"
    "${PROJECT_BINARY_DIR}/opti_config.h"
  )

# Install Config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/optiConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/optiConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Export from the build tree
export(EXPORT opti-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/optiTargets.cmake
    NAMESPACE opti::
       )

export(PACKAGE opti)

# Sub Directories
add_subdirectory(examples)
