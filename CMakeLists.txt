cmake_minimum_required(VERSION 3.7)
project( teera_lab2_homework )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_PREFIX_PATH "/usr/local/opt/opencv3/share/OpenCV")
set(OpenCV_INCLUDE_DIRS "/usr/local/opt/opencv3/include")
set(OpenCV_LIBS "/usr/local/opt/opencv3/lib")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( teera_lab2_homework teera_lab2_homework.cpp )
target_link_libraries( teera_lab2_homework ${OpenCV_LIBS} )
