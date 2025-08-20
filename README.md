# FR3 MuJoCo ROS1

## 1) Overview
A minimal ROS1 (Python) stack with a MuJoCo-based simulator, an example controller, and shared message definitions.

- `mujoco_ros_sim`: Publishes physics step results (state/sensors/images) and subscribes to control inputs.
- `mujoco_ros_sim_msgs`: Shared message package used by the simulator and controller.
- `fr3_controller`: Subscribes to joint/image data, republishes the `hand_eye` image, and publishes sine-wave demo control inputs.

---

## 2) Package Descriptions

### A) `mujoco_ros_sim`
Loads/steps a MuJoCo model and publishes:  
`mujoco_ros_sim/joint_dict` (JointDict), `mujoco_ros_sim/sensor_dict` (SensorDict), `mujoco_ros_sim/image_dict` (ImageDict), and `joint_states` (sensor_msgs/JointState).  
Subscribes to `mujoco_ros_sim/ctrl_dict` (CtrlDict) and applies control inputs to actuators.

### B) `mujoco_ros_sim_msgs`
Shared messages between simulator and controller:  
`NamedFloat64Array`, `JointDict`, `SensorDict`, `NamedImage`, `ImageDict`, `CtrlDict`.

### C) `fr3_controller`
Subscribes to `JointDict` and prints FR3’s 7 joints plus all joints, consumes `ImageDict` to republish the `hand_eye` image as `sensor_msgs/Image`, and publishes a per-joint sine control input.

---

## 3) Build
```bash
cd ~/catkin_ws/src
git clone https://github.com/JunHeonYoon/fr3_mujoco_ros1.git
cd ..
catkin build
source devel/setup.bash
```

---

## 4) Run
```bash
# 1) Launch MuJoCo simulator only
roslaunch mujoco_ros_sim mujoco_sim.launch

# 2) Launch MuJoCo + FR3 Controller
roslaunch fr3_controller fr3_controller.launch
```
---
## 5) Controller Architecture

The controller exposes the following hooks; place your custom code inside each function.

- `init()` 
Configure ROS parameters, publishers, and subscribers.

- `starting()` 
One-time initialization at start (reference time, internal states, etc.).

- `updateState(msg: JointDict)` 
Receive/store/log joint states (e.g., print FR3’s 7 joints and all joints).

- `updateImage(msg: ImageDict)`
Receive/store/log image data (e.g., recieve hand_eye camera image and republish to hand_eye/image_raw).

- `compute(t: float)`
Compute control input at time t (e.g., sine-wave demo) and publish to mujoco_ros_sim/ctrl_dict.

