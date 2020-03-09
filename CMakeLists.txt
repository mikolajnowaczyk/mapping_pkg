cmake_minimum_required(VERSION 2.8.3)
project(mapping_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

message(PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR} )
message(CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR} )

set(Mapping_DIR "${PROJECT_SOURCE_DIR}/cmake/packages")
set(Defs_DIR "${PROJECT_SOURCE_DIR}/cmake/packages")
set(Filtering_DIR "${PROJECT_SOURCE_DIR}/cmake/packages")
set(ImageProcessing_DIR "${PROJECT_SOURCE_DIR}/cmake/packages")
set(Utilities_DIR "${PROJECT_SOURCE_DIR}/cmake/packages")
set(Visualizer_DIR "${PROJECT_SOURCE_DIR}/cmake/packages")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  rviz
  Mapping
  Defs
  Utilities
  ImageProcessing
  Filtering
  Visualizer
)

###############################################################################
#
# OpenCV library
#
###############################################################################

set(OpenCV_DIR ~/Libs/opencv/build)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(CMAKE_BUILD_TYPE Debug)

###############################################################################
#
# Eigen3 library
#
###############################################################################
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(-Wno-deprecated -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

###############################################################################
#
# External Octree library
#
###############################################################################

include_directories(${MAPPING_INCLUDE_DIRS}/..)
file(GLOB OCTREE_HEADERS
    "${CMAKE_SOURCE_DIR}/include/octree/*.h"
    "${CMAKE_SOURCE_DIR}/include/octree/*.tcc"
)
INSTALL(FILES ${OCTREE_HEADERS} DESTINATION include/octree)

###############################################################################
#
# PCL
#
###############################################################################

find_package(PCL 1.8 REQUIRED)
set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#QGLViewer library
FIND_PACKAGE(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
link_directories(${OPENGL_LIBRARY})

#QGLViewer library
FIND_PACKAGE(QGLViewer REQUIRED)
include_directories(${QGLVIEWER_INCLUDE_DIR})
include_directories(${QT_INCLUDES})
link_directories(${QGLVIEWER_LIBRARY})

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs ellipsoid_msgs rviz cv_bridge
  DEPENDS
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${MAPPING_INCLUDE_DIRS}
)
link_directories(${catkin_LIBRARIES} ${MAPPING_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(mapping_node src/mapping_pkg_node.cpp)

add_dependencies(mapping_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(mapping_node
   ${catkin_LIBRARIES} QtXml Mapping Filtering ImageProcessing Utilities Defs Visualizer tinyxml2 ${QGLVIEWER_LIBRARY} ${QT_LIBRARIES} ${QT_LIBRARIES} QtCore  QtOpenGL QtGui ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

