cmake_minimum_required(VERSION 3.6)
project(FullConnectedNeuralNet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp FCLayer.cpp FCLayer.h PoolLayer.cpp PoolLayer.h ConvLayer.cpp ConvLayer.h Layer.h MNeuralNet.cpp MNeuralNet.h)
add_executable(NN_neuron ${SOURCE_FILES})
