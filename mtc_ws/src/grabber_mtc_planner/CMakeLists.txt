cmake_minimum_required(VERSION 3.16)
project(grabber_mtc_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(moveit_task_constructor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(plan_shelf_transfer
  src/plan_shelf_transfer.cpp
  src/scenario.cpp
)
target_include_directories(plan_shelf_transfer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
ament_target_dependencies(plan_shelf_transfer
  ament_index_cpp
  rclcpp
  sensor_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
  builtin_interfaces
  geometry_msgs
  shape_msgs
  trajectory_msgs
  moveit_msgs
  tf2_eigen
  tf2_geometry_msgs
)
# yaml-cpp exports an imported target on Humble; fall back to the plain library
# name on installations that only provide YAML_CPP_LIBRARIES.
if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(plan_shelf_transfer yaml-cpp::yaml-cpp)
else()
  target_link_libraries(plan_shelf_transfer ${YAML_CPP_LIBRARIES})
endif()

install(TARGETS plan_shelf_transfer DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/gate_c_review.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch scenarios DESTINATION share/${PROJECT_NAME})

ament_package()
