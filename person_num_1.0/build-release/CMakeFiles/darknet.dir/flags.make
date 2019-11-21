# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# compile C with /usr/bin/cc
# compile CUDA with /usr/local/cuda/bin/nvcc
# compile CXX with /usr/bin/c++
C_FLAGS = -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings   

C_DEFINES = -DCUDNN -DCUDNN_HALF -DGPU -DUSE_CMAKE_LIBS

C_INCLUDES = -I/media/D/train_code/darknet-master/include -I/media/D/train_code/darknet-master/src -I/media/D/train_code/darknet-master/3rdparty/stb/include -isystem /usr/local/cuda/include 

CUDA_FLAGS = -gencode arch=compute_75,code=sm_75 --compiler-options " -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings -DGPU -DCUDNN -fPIC -fopenmp -Ofast "   

CUDA_DEFINES = -DCUDNN -DCUDNN_HALF -DGPU -DUSE_CMAKE_LIBS

CUDA_INCLUDES = -I/media/D/train_code/darknet-master/include -I/media/D/train_code/darknet-master/src -I/media/D/train_code/darknet-master/3rdparty/stb/include -isystem=/usr/local/cuda/include 

CXX_FLAGS = -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings    -std=gnu++11

CXX_DEFINES = -DCUDNN -DCUDNN_HALF -DGPU -DUSE_CMAKE_LIBS

CXX_INCLUDES = -I/media/D/train_code/darknet-master/include -I/media/D/train_code/darknet-master/src -I/media/D/train_code/darknet-master/3rdparty/stb/include -isystem /usr/local/cuda/include 

