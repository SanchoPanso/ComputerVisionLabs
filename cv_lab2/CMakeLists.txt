cmake_minimum_required(VERSION 3.20)
project(cv_lab2)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cv_lab2 lab2.cpp)
target_link_libraries(cv_lab2 ${OpenCV_LIBS})
