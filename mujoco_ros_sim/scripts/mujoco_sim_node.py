#!/usr/bin/env python3
import time
import threading
from array import array

import numpy as np
import rospy

from sensor_msgs.msg import JointState, Image
from mujoco_ros_sim_msgs.msg import JointDict, SensorDict, CtrlDict, ImageDict, NamedImage

import mujoco
import mujoco.viewer

# utils: load_mj_model, print_table, to_NamedFloat64ArrayMsg
from mujoco_ros_sim.utils import load_mj_model, print_table, to_NamedFloat64ArrayMsg


def ros_time_from_float(t: float) -> rospy.Time:
    secs = int(t)
    nsecs = int(round((t - secs) * 1e9))
    return rospy.Time(secs, nsecs)


class MujocoSimNode:
    def __init__(self):
        self._mj_lock   = threading.Lock()   # MuJoCo step/render 보호
        self.state_lock = threading.Lock()   # 제어명령(ctrl) 보호
        self._ready = False

        rospy.init_node('mujoco_sim_node')
        robot_name = rospy.get_param('~robot_name')
        self.camera_fps = float(rospy.get_param('~camera_fps', 60.0))

        self.joint_dict_pub  = rospy.Publisher('mujoco_ros_sim/joint_dict',  JointDict,  queue_size=1)
        self.sensor_dict_pub = rospy.Publisher('mujoco_ros_sim/sensor_dict', SensorDict, queue_size=1)
        self.image_pub       = rospy.Publisher('mujoco_ros_sim/image_dict',  ImageDict,  queue_size=1)
        self.joint_state_pub = rospy.Publisher('joint_states',               JointState, queue_size=10)

        self._ctrl_sub = rospy.Subscriber('mujoco_ros_sim/ctrl_dict', CtrlDict, self.sub_ctrl_cb, queue_size=1)

        self.mj_model = load_mj_model(robot_name)
        rospy.loginfo("\n\033[1;34m" + print_table(robot_name, self.mj_model) + "\033[0m")
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.dt = float(self.mj_model.opt.timestep)

        self.viewer_fps = 60.0
        try:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data,
                                                       show_left_ui=False, show_right_ui=False)
        except Exception as e:
            rospy.logwarn(f"[viewer] failed to launch passive viewer: {e}")
            self.viewer = None

        self.joint_dict = {"joint_names": [], "jname_to_jid": {}}
        for i in range(self.mj_model.njnt):
            name_adr = self.mj_model.name_jntadr[i]
            jname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            if not jname:
                continue
            self.joint_dict["joint_names"].append(jname)
            self.joint_dict["jname_to_jid"][jname] = i

        self.actuator_dict = {"actuator_names": [], "aname_to_aid": {}}
        for i in range(self.mj_model.nu):
            name_adr = self.mj_model.name_actuatoradr[i]
            aname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            self.actuator_dict["actuator_names"].append(aname)
            self.actuator_dict["aname_to_aid"][aname] = i

        self.sensor_dict = {"sensor_names": [], "sname_to_sid": {}, "sname_to_sdim": {}}
        for i in range(self.mj_model.nsensor):
            name_adr = self.mj_model.name_sensoradr[i]
            sname = self.mj_model.names[name_adr:].split(b'\x00', 1)[0].decode('utf-8')
            self.sensor_dict["sensor_names"].append(sname)
            self.sensor_dict["sname_to_sid"][sname] = self.mj_model.sensor_adr[i]
            self.sensor_dict["sname_to_sdim"][sname] = self.mj_model.sensor_dim[i]

        self.camera_dict = {"cam_names": [], "cname_to_cid": {}, "resolution": (0, 0)}
        for i in range(self.mj_model.ncam):
            cadr  = self.mj_model.name_camadr[i]
            cname = self.mj_model.names[cadr:].split(b'\x00', 1)[0].decode('utf-8')
            if cname in ("", "free"):
                continue
            self.camera_dict["cam_names"].append(cname)
            self.camera_dict["cname_to_cid"][cname] = cadr

        offw = int(self.mj_model.vis.global_.offwidth)  or 640
        offh = int(self.mj_model.vis.global_.offheight) or 480
        self.camera_dict["resolution"] = (offw, offh)

        self._j_slices = []
        for jname in self.joint_dict["joint_names"]:
            jid   = self.joint_dict["jname_to_jid"][jname]
            idx_q = self.mj_model.jnt_qposadr[jid]
            idx_v = self.mj_model.jnt_dofadr[jid]
            next_q = (self.mj_model.jnt_qposadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nq)
            next_v = (self.mj_model.jnt_dofadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nv)
            nq = next_q - idx_q
            nv = next_v - idx_v
            self._j_slices.append((idx_q, nq, idx_v, nv, jname))

        self._s_slices = []
        for sname in self.sensor_dict["sensor_names"]:
            idx_s = self.sensor_dict["sname_to_sid"][sname]
            ns    = self.sensor_dict["sname_to_sdim"][sname]
            self._s_slices.append((idx_s, ns, sname))

        self._pos_msgs    = [to_NamedFloat64ArrayMsg(jn, np.zeros(nq)) for (_, nq,  _, _,  jn) in self._j_slices]
        self._vel_msgs    = [to_NamedFloat64ArrayMsg(jn, np.zeros(nv)) for (_,  _,  _, nv, jn) in self._j_slices]
        self._tau_msgs    = [to_NamedFloat64ArrayMsg(jn, np.zeros(nv)) for (_,  _,  _, nv, jn) in self._j_slices]
        self._sensor_msgs = [to_NamedFloat64ArrayMsg(sn, np.zeros(ns)) for (_, ns, sn) in self._s_slices]

        self._pos_lists    = [m.value.data for m in self._pos_msgs]
        self._vel_lists    = [m.value.data for m in self._vel_msgs]
        self._tau_lists    = [m.value.data for m in self._tau_msgs]
        self._sensor_lists = [m.value.data for m in self._sensor_msgs]

        self._imgdict_msg = ImageDict()
        self._imgdict_msg.images = []
        self._cam_buf_views = {}
        self._cam_named = {}

        if len(self.camera_dict["cam_names"]) > 0:
            W, H = self.camera_dict["resolution"]
            for cname in self.camera_dict["cam_names"]:
                img = Image()
                img.width, img.height = W, H
                img.encoding = "rgb8"
                img.is_bigendian = 0
                img.step = W * 3
                img.data = bytearray(W * H * 3)
                self._cam_buf_views[cname] = np.frombuffer(img.data, dtype=np.uint8)
                ni = NamedImage()
                ni.name, ni.image = cname, img
                self._imgdict_msg.images.append(ni)
                self._cam_named[cname] = ni

        self._render_data = mujoco.MjData(self.mj_model)

        self._sim_timer    = rospy.Timer(rospy.Duration.from_sec(self.dt),               self.sim_loop)
        self._viewer_timer = rospy.Timer(rospy.Duration.from_sec(1.0 / self.viewer_fps), self.viewer_loop)
        self._joint_timer  = rospy.Timer(rospy.Duration.from_sec(0.01),                  self.pub_joint_cb)

        self._cam_thread = None
        if len(self.camera_dict["cam_names"]) > 0:
            self._cam_thread = threading.Thread(target=self.camera_thread, daemon=True)
            self._cam_thread.start()

        rospy.on_shutdown(self.on_shutdown)

        self._ready = True

    def sim_loop(self, _event):
        t0 = time.perf_counter()
        with self._mj_lock:
            mujoco.mj_step(self.mj_model, self.mj_data)
        t1 = time.perf_counter()

        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        qfrc = self.mj_data.qfrc_applied
        sens = self.mj_data.sensordata

        for i, (idx_q, nq, idx_v, nv, _) in enumerate(self._j_slices):
            self._pos_lists[i][:] = array('d', qpos[idx_q: idx_q + nq])
            self._vel_lists[i][:] = array('d', qvel[idx_v: idx_v + nv])
            self._tau_lists[i][:] = array('d', qfrc[idx_v: idx_v + nv])
        for i, (idx_s, ns, _) in enumerate(self._s_slices):
            self._sensor_lists[i][:] = array('d', sens[idx_s: idx_s + ns])
        t2 = time.perf_counter()

        now = rospy.Time.now()

        joint_dict_msg = JointDict()
        joint_dict_msg.header.stamp = now
        joint_dict_msg.sim_time = ros_time_from_float(self.mj_data.time)
        joint_dict_msg.positions  = self._pos_msgs
        joint_dict_msg.velocities = self._vel_msgs
        joint_dict_msg.torques    = self._tau_msgs
        self.joint_dict_pub.publish(joint_dict_msg)

        sensor_dict_msg = SensorDict()
        sensor_dict_msg.header.stamp = now
        sensor_dict_msg.sim_time = ros_time_from_float(self.mj_data.time)
        sensor_dict_msg.sensors = self._sensor_msgs
        self.sensor_dict_pub.publish(sensor_dict_msg)
        t3 = time.perf_counter()

    def viewer_loop(self, _event):
        try:
            with self._mj_lock:
                if self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
        except Exception:
            pass

    def pub_joint_cb(self, _event):
        joint_names = self.joint_dict["joint_names"]
        if not joint_names:
            return

        positions, velocities = [], []
        for jname in joint_names:
            jid   = self.joint_dict["jname_to_jid"][jname]
            idx_q = self.mj_model.jnt_qposadr[jid]
            idx_v = self.mj_model.jnt_dofadr[jid]
            next_q = (self.mj_model.jnt_qposadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nq)
            next_v = (self.mj_model.jnt_dofadr[jid + 1] if jid + 1 < self.mj_model.njnt else self.mj_model.nv)
            nq = next_q - idx_q
            nv = next_v - idx_v
            positions.extend(self.mj_data.qpos[idx_q: idx_q + nq].tolist())
            velocities.extend(self.mj_data.qvel[idx_v: idx_v + nv].tolist())

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_names
        msg.position = positions
        msg.velocity = velocities
        self.joint_state_pub.publish(msg)

    def camera_thread(self):
        try:
            W, H = self.camera_dict["resolution"]
            self.cam_renderer = mujoco.Renderer(self.mj_model, width=W, height=H)
            self.cam_renderer.disable_depth_rendering()
        except Exception as e:
            rospy.logerr(f"[camera_thread] failed to create renderer: {e}")
            return

        period = 1.0 / self.camera_fps

        try:
            while not rospy.is_shutdown():
                t0 = time.perf_counter()

                with self._mj_lock:
                    rd = self._render_data
                    sd = self.mj_data
                    rd.qpos[:] = sd.qpos
                    rd.qvel[:] = sd.qvel
                    if self.mj_model.nu:
                        rd.act[:] = sd.act
                    if self.mj_model.nmocap:
                        rd.mocap_pos[:]  = sd.mocap_pos
                        rd.mocap_quat[:] = sd.mocap_quat
                    rd.time = sd.time

                mujoco.mj_forward(self.mj_model, self._render_data)

                stamp = rospy.Time.now()

                for cname in self.camera_dict["cam_names"]:
                    try:
                        self.cam_renderer.update_scene(self._render_data, camera=cname)
                        rgb = self.cam_renderer.render()  # (H, W, 3) uint8
                        self._cam_buf_views[cname][:] = rgb.reshape(-1)
                        self._cam_named[cname].image.header.stamp = stamp
                    except Exception as e:
                        rospy.logwarn(f"[camera_thread] {cname}: {e}")

                self._imgdict_msg.header.stamp = stamp
                self._imgdict_msg.sim_time = ros_time_from_float(self._render_data.time)
                self.image_pub.publish(self._imgdict_msg)

                loop_dt = time.perf_counter() - t0
                if loop_dt < period:
                    time.sleep(period - loop_dt)
        finally:
            try:
                self.cam_renderer = None
            except Exception:
                pass

    def sub_ctrl_cb(self, msg: CtrlDict):
        if not getattr(self, '_ready', False):
            return

        ctrl_command = {}
        for item in msg.commands:
            data = list(item.value.data)
            ctrl_command[item.name] = data if len(data) > 1 else (data[0] if data else 0.0)

        # 이름 검증 + 쓰기 (state_lock + mj_lock 동시 확보)
        with self.state_lock, self._mj_lock:
            for name, cmd in ctrl_command.items():
                aid = self.actuator_dict["aname_to_aid"].get(name, None)
                if aid is None:
                    rospy.logwarn_throttle(2.0, "[ctrl] unknown actuator name: %s", name)
                    continue
                self.mj_data.ctrl[aid] = float(cmd)

    def on_shutdown(self):
        if self._cam_thread is not None and self._cam_thread.is_alive():
            self._cam_thread.join(timeout=1.0)
        try:
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.close()
        except Exception:
            pass


if __name__ == "__main__":
    node = MujocoSimNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
