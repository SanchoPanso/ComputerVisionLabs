cmake_minimum_required(VERSION 3.0.0)
project(cv_labs VERSION 0.1.0)

include(CTest)
enable_testing()

#set(OpenCV_DIR "D:\\lib\\opencv_with_contrib\\install\\opencv")    # VS
set(OpenCV_DIR "D:\\lib\\opencv_4.7.0\\build_mingw\\install")    # MinGW
message(${OpenCV_DIR})
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})


# # cv_lab1
add_executable(cv_lab1 cv_lab1/main.cpp)
target_link_libraries(cv_lab1 ${OpenCV_LIBS})

# cv_lab2
add_executable(cv_lab2 cv_lab2/main.cpp cv_lab2/TickMeter.cpp)
target_link_libraries(cv_lab2 ${OpenCV_LIBS})

# cv_lab3
add_executable(cv_lab3 cv_lab3/main.cpp cv_lab3/cvDirectory.cpp)
target_link_libraries(cv_lab3 ${OpenCV_LIBS})
add_executable(begunok cv_lab3/begunok.cpp cv_lab3/cvDirectory.cpp)
target_link_libraries(begunok ${OpenCV_LIBS})

# cv_lab4
add_executable(cv_lab4 cv_lab4/main.cpp)
target_link_libraries(cv_lab4 ${OpenCV_LIBS})

# cv_lab5
add_executable(cv_lab5 cv_lab5/main.cpp)
target_link_libraries(cv_lab5 ${OpenCV_LIBS})
add_executable(calibrate_camera cv_lab5/calibrate_camera.cpp)
target_link_libraries(calibrate_camera ${OpenCV_LIBS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
