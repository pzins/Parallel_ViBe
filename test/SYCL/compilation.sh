#!/usr/bin/bash
FILENAME="vibe"


SYCL_ROOT="/usr/local/computecpp"
RUNTIME_INCLUDE_PATH="$SYCL_ROOT/include"
RUNTIME_LIB="$SYCL_ROOT/lib/libComputeCpp.so"

CXX=g++

DEVICE_CXX="$SYCL_ROOT/bin/compute++"
DEVICE_FLAGS="-O3 -sycl -std=c++11 -emit-llvm -I$RUNTIME_INCLUDE_PATH"

CXX_FLAGS="-O3 -std=c++11 -I$RUNTIME_INCLUDE_PATH -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio -lopencv_videostab"
OPENCL_LIB_DIR=/opt/rocm/opencl/lib/x86_64
LD_FLAGS="-L$OPENCL_LIB_DIR -lOpenCL -pthread"

$SYCL_ROOT/bin/compute++ $DEVICE_FLAGS -c $FILENAME.cpp -o $FILENAME.bc
#$CXX $CXX_FLAGS -include $FILENAME.sycl $FILENAME.cpp $RUNTIME_LIB $LD_FLAGS -o $FILENAME
g++ -include $FILENAME.sycl vibe.cpp -I /usr/local/computecpp/include/ -L /usr/local/computecpp/lib/ -lComputeCpp -lOpenCL -std=c++11 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

