cmake_minimum_required(VERSION 3.0.2)
project(mujoco_ros_sim_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  message_generation
)

## 메시지 파일 나열
add_message_files(
  DIRECTORY msg
  FILES
    NamedFloat64Array.msg
    JointDict.msg
    SensorDict.msg
    NamedImage.msg
    ImageDict.msg
    CtrlDict.msg
)

## 메시지 생성
generate_messages(
  DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs
)
