#!/usr/bin/bash
FILENAME="vector"
# FILENAME="hello_world"


SYCL_ROOT="/usr/local/computecpp"
RUNTIME_INCLUDE_PATH="$SYCL_ROOT/include"
RUNTIME_LIB="$SYCL_ROOT/lib/libComputeCpp.so"

CXX=g++

DEVICE_CXX="$SYCL_ROOT/bin/compute++"
DEVICE_FLAGS="-O3 -sycl -std=c++11 -emit-llvm -I$RUNTIME_INCLUDE_PATH"

CXX_FLAGS="-O3 -std=c++11 -I$RUNTIME_INCLUDE_PATH"
OPENCL_LIB_DIR=/opt/rocm/opencl/lib/x86_64
LD_FLAGS="-L$OPENCL_LIB_DIR -lOpenCL -pthread"

$SYCL_ROOT/bin/compute++ $DEVICE_FLAGS -c $FILENAME.cpp -o $FILENAME.bc
$CXX $CXX_FLAGS -include $FILENAME.sycl $FILENAME.cpp $RUNTIME_LIB $LD_FLAGS -o $FILENAME
# $CXX $CXX_FLAGS $FILENAME.o -llttng-ust -ldl $RUNTIME_LIB $LD_FLAGS -o $FILENAME
