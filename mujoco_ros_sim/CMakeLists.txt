cmake_minimum_required(VERSION 3.0.2)
project(mujoco_ros_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  mujoco_ros_sim_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs mujoco_ros_sim_msgs
)

# Python 실행 스크립트 설치
catkin_install_python(
  PROGRAMS scripts/mujoco_sim_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# launch 설치 (선택)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY franka_fr3
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)