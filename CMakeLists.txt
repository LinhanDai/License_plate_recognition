cmake_minimum_required(VERSION 3.1) 
project(ocr_sys VERSION 1.0)
enable_language(CUDA)

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

#include
include_directories(${PROJECT_SOURCE_DIR}/include /usr/include/freetype2)

#cuda
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()
find_package(CUDA REQUIRED)

#opencv
find_package(OpenCV REQUIRED)

#detector lib (cuda)
file(GLOB_RECURSE sources modules/*.hpp modules/*.cpp modules/*.h modules/*.cu extra/*.h src/*.cpp)
add_library(ocr SHARED ${sources})
target_include_directories(ocr PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(ocr nvinfer nvinfer_plugin nvcaffe_parser "stdc++fs")

#sample
add_executable(ocr_sys src/demo.cpp)
target_include_directories(ocr_sys PRIVATE modules/ extra/)
target_link_libraries(ocr_sys ocr ${OpenCV_LIBS} freetype)