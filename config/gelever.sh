#!/bin/sh
# Copyright (c) 2018, Stephan Gelever

BASE_DIR=${PWD}
BUILD_DIR=${BASE_DIR}/build
EXTRA_ARGS=$@

mkdir -p $BUILD_DIR
cd $BUILD_DIR

rm -rf CMakeCache.txt
rm -rf CMakeFiles

METIS_DIR=~/metis
HYPRE_DIR=~/hypre
SUITESPARSE_DIR=~/SuiteSparse
BUILD_TYPE=Debug

FC=/usr/bin/x86_64-linux-gnu-gfortran-7 CC=mpicc CXX=mpic++ cmake \
    -DHypre_INC_DIR=${HYPRE_DIR}/include \
    -DHypre_LIB_DIR=${HYPRE_DIR}/lib \
    -DSUITESPARSE_INCLUDE_DIR_HINTS=${SUITESPARSE_DIR}/include \
    -DSUITESPARSE_LIBRARY_DIR_HINTS=${SUITESPARSE_DIR}/lib \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    ${BASE_DIR} \
    ${EXTRA_ARGS}

