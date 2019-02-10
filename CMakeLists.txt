cmake_minimum_required(VERSION 3.13)
project(Code)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-fopenmp")
add_executable(Code subString_4.cpp)