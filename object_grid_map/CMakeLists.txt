cmake_minimum_required(VERSION 3.0.2)
project(object_grid_map)

IF (NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
		roscpp#
		rospy
		pcl_ros#
		pcl_conversions#
		eigen_conversions #
		cv_bridge   #
		sensor_msgs#
		image_transport#
		nav_msgs#
		)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED QUIET)

catkin_package(

)

include_directories(
		include
		${catkin_INCLUDE_DIRS}
		${PCL_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
)

link_directories(
		include
		${OpenCV_LIBRARY_DIRS}
		${PCL_LIBRARY_DIRS}
)

file(GLOB_RECURSE ALL_SRC "src/*.cc")
file(GLOB_RECURSE ALL_SRC_CPP "src/*.cpp")

add_executable(objseg_gridclass node/objseg_grid_class.cpp ${ALL_SRC} ${ALL_SRC_CPP})
add_dependencies(objseg_gridclass ${catkin_EXPORTED_TARGETS})
target_link_libraries(objseg_gridclass ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES} ${PCL_LIBRARIES})

