#####################
## cmake config below
cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)
set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/bin")
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE INTERNAL "")
set(DATA_ROOT_PATH "${CMAKE_SOURCE_DIR}/data" CACHE INTERNAL "")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(COMPUTECPP_PACKAGE_ROOT_DIR /usr/local/computecpp)
########################
## solution config below
project(inf6803_h2017)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>)
find_package(OpenCV 3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


message(/usr/local/computecpp/include)
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
mark_as_advanced(OPENCV_FOUND OpenCV_FOUND OpenCV_DIR)
add_definitions(-march=native)
add_definitions(-Wall)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/include"
)

#############################
## TP1 projects configs below
include_directories("include")
configure_file(
    "${CMAKE_SOURCE_DIR}/include/common.hpp.conf"
    "${CMAKE_BINARY_DIR}/include/common.hpp"
)
#add_executable(tp1_main
 #   "src/main.cpp"
  #  "src/vibe_cpu.cpp"
   # "src/kernel.cl"
    #"include/common.hpp"
#)
#message(${OpenCV_LIBRARIES})
include(FindComputeCpp)

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

add_executable(tp1_main     
	"src/main.cpp"
    "src/vibe_sycl.cpp"
    "src/kernel.cl"
    "include/common.hpp"  )
target_compile_options(tp1_main PUBLIC -std=c++11 -Wall)

add_sycl_to_target(tp1_main  ${CMAKE_CURRENT_BINARY_DIR}
                    /home/pierre/OpenCL-SYCL/Parallel_ViBe/src/vibe_sycl.cpp)

target_link_libraries(tp1_main ${OpenCV_LIBRARIES} OpenCL)
