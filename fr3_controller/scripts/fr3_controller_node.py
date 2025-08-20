#!/usr/bin/env python3
import numpy as np
import math
import time
import threading

import rospy
from std_msgs.msg import Header, Float64MultiArray
from sensor_msgs.msg import Image
from mujoco_ros_sim_msgs.msg import CtrlDict, NamedFloat64Array, JointDict, ImageDict

# 제어할 조인트 이름들 (mujoco actuator 이름과 동일해야 함)
JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]  # fr3_joint1 ... fr3_joint7

class FR3ControllerNode:
    def __init__(self):
        # 공용 상태
        self._lock = threading.Lock()
        self._last_print = 0.0
        self._q  = {n: float('nan') for n in JOINT_NAMES}
        self._dq = {n: float('nan') for n in JOINT_NAMES}
        self._t0 = None  # starting()에서 설정

        # 분리된 초기화 단계
        self.init()

    # =========================
    # 1) 초기화 훅
    # =========================
    def init(self):
        """ROS 노드/파라미터/퍼블리셔/구독자 설정. (→ 필요시 사용자 훅 추가)"""
        rospy.init_node("fr3_controller")

        # 파라미터
        self.pub_rate_hz = float(rospy.get_param("~pub_rate", 200.0))  # CtrlDict publish 주기
        self.amp         = float(rospy.get_param("~amp", 0.2))         # 사인 진폭
        self.freq        = float(rospy.get_param("~freq", 0.5))        # 사인 주파수

        # 퍼블리셔
        self._ctrl_pub = rospy.Publisher("mujoco_ros_sim/ctrl_dict", CtrlDict, queue_size=1)
        self._img_pub  = rospy.Publisher("hand_eye/image_raw", Image, queue_size=1)  # 재퍼블리시

        # 구독자 (사용자 코드: 다른 토픽 구독 추가 가능)
        self._joint_sub = rospy.Subscriber("mujoco_ros_sim/joint_dict", JointDict,
                                           self.updateState, queue_size=10)
        self._image_sub = rospy.Subscriber("mujoco_ros_sim/image_dict", ImageDict,
                                           self.updateImage, queue_size=5)

        rospy.loginfo("[fr3_controller] init done. pub_rate=%.1fHz, amp=%.3f, freq=%.3f",
                      self.pub_rate_hz, self.amp, self.freq)

    # =========================
    # 2) 시작 훅
    # =========================
    def starting(self):
        """컨트롤 시작 직전 한 번 실행. (→ 필요시 사용자 훅 추가)"""
        self._t0 = time.time()
        rospy.loginfo("[fr3_controller] starting at t0=%.3f", self._t0)


    # =========================
    # 3) 상태 업데이트 콜백
    # =========================
    def updateState(self, msg: JointDict):
        """JointDict 콜백. 이름별 q, dq 갱신 및 상태 프린트. (→ 필요시 사용자 훅 추가)"""
        # positions/velocities 배열을 이름 기반으로 딕셔너리로 변환
        pos_map = {item.name: item for item in msg.positions}
        vel_map = {item.name: item for item in msg.velocities}

        with self._lock:
            for n in JOINT_NAMES:
                # 각 조인트는 보통 스칼라 1개(dof=1). 다자유도면 첫 원소만 사용.
                self._q[n]  = float(pos_map[n].value.data[0]) if (n in pos_map and pos_map[n].value.data) else float('nan')
                self._dq[n] = float(vel_map[n].value.data[0]) if (n in vel_map and vel_map[n].value.data) else float('nan')

        # --- 사용자 훅(옵션): 예) 관측치 필터링/추정기 업데이트 ---
        # self._estimator.update(self._q, self._dq)

        # 2 Hz로 상태 출력 (예시: FR3 7개 + 전체 조인트)
        now = time.time()
        if now - self._last_print > 0.5:
            self._last_print = now
            self._print_selected()  # FR3 7개
            self._print_all(msg)    # 전체

    def _print_selected(self):
        try:
            q  = [self._q[n]  for n in JOINT_NAMES]
            dq = [self._dq[n] for n in JOINT_NAMES]
            rospy.loginfo_throttle(0.5,
                "[FR3] q = [%.3f %.3f %.3f %.3f %.3f %.3f %.3f], dq = [%.3f %.3f %.3f %.3f %.3f %.3f %.3f]",
                *q, *dq)
        except Exception:
            pass

    def _print_all(self, msg: JointDict):
        try:
            # 이름 -> 전체 배열 매핑
            pos_map = {it.name: list(it.value.data) for it in msg.positions}
            vel_map = {it.name: list(it.value.data) for it in msg.velocities}
            names = sorted(set(pos_map.keys()) | set(vel_map.keys()))

            lines = ["[ALL]"]
            for n in names:
                q_arr  = np.array(pos_map.get(n, []), dtype=float)
                dq_arr = np.array(vel_map.get(n, []), dtype=float)
                lines.append(f"{n}: q={q_arr}, dq={dq_arr}")  # 이름: numpy array 형태

            rospy.loginfo_throttle(0.5, "\n".join(lines))
        except Exception:
            pass

    # =========================
    # 4) 이미지 업데이트 콜백
    # =========================
    def updateImage(self, msg: ImageDict):
        """ImageDict 콜백. hand_eye 이미지를 찾아 재퍼블리시. (→ 필요시 사용자 훅 추가)"""
        for ni in msg.images:
            if ni.name == "hand_eye":
                # 원본 header를 그대로 유지 (타임스탬프 포함)
                self._img_pub.publish(ni.image)

                # --- 사용자 훅(옵션): 예) OpenCV 처리 후 별도 토픽으로 퍼블리시 ---
                # cv_img = cv2.cvtColor(np.frombuffer(ni.image.data, dtype=np.uint8).reshape(H, W, 3), cv2.COLOR_RGB2BGR)
                # self._img_pub_processed.publish(self._bridge.cv2_to_imgmsg(cv_img, encoding="bgr8"))
                return

        rospy.logwarn_throttle(5.0, "[fr3_controller] 'hand_eye' image not found in ImageDict")

    # =========================
    # 5) 제어 계산 훅
    # =========================
    def compute(self, t: float):
        """시각 t에서 제어입력 계산 & 퍼블리시. (→ 필요시 사용자 훅으로 교체)"""
        # --- 예시 구현: 각 관절에 위상차 사인 제어 ---
        cmds = []
        for k, name in enumerate(JOINT_NAMES):
            phase = k * (math.pi / 8.0)
            u = self.amp * math.sin(2.0 * math.pi * self.freq * t + phase)

            nfa = NamedFloat64Array()
            nfa.name = name
            f = Float64MultiArray()
            f.data = [u]  # 단일 스칼라
            nfa.value = f
            cmds.append(nfa)

        msg = CtrlDict()
        msg.header = Header(stamp=rospy.Time.now())
        msg.commands = cmds
        self._ctrl_pub.publish(msg)

        # --- 사용자 훅(옵션): 예) QP/MPC 결과 퍼블리시 ---
        # msg = self._qp.solve(self._q, self._dq, t)
        # self._ctrl_pub.publish(msg)

    # =========================
    # 메인 루프
    # =========================
    def spin(self):
        self.starting()
        rate = rospy.Rate(self.pub_rate_hz)
        while not rospy.is_shutdown():
            t = time.time() - self._t0 if self._t0 is not None else 0.0
            self.compute(t)
            rate.sleep()


if __name__ == "__main__":
    node = FR3ControllerNode()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
