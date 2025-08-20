cmake_minimum_required(VERSION 3.0.2)
project(fr3_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  mujoco_ros_sim_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs mujoco_ros_sim_msgs
)

catkin_install_python(
  PROGRAMS scripts/fr3_controller_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
