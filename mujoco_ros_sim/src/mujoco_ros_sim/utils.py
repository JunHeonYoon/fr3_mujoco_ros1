import os
import textwrap
import numpy as np

import mujoco

# ROS1 runtime에서 import (메시지 타입은 msgs 패키지에 존재해야 함)
from std_msgs.msg import Float64MultiArray
from mujoco_ros_sim_msgs.msg import NamedFloat64Array

def load_mj_model(robot_name: str) -> mujoco.MjModel:
    xml_path = robot_name
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"MJCF not found: {xml_path}")
    return mujoco.MjModel.from_xml_path(xml_path)

def print_table(robot_name: str, m: mujoco.MjModel) -> str:
    jt_enum = mujoco.mjtJoint
    enum2name = {getattr(jt_enum, a): a[5:].title() for a in dir(jt_enum) if a.startswith("mjJNT_")}

    lines = []
    lines.append("=================================================================")
    lines.append("=================================================================")
    lines.append(f"MuJoCo Model Information: {robot_name}")
    lines.append(" id | name                 | type   | nq | nv | idx_q | idx_v")
    lines.append("----+----------------------+--------+----+----+-------+------")

    for jid in range(m.njnt):
        adr  = m.name_jntadr[jid]
        name = m.names[adr:].split(b'\x00', 1)[0].decode()
        if not name:
            continue
        jtype    = int(m.jnt_type[jid])
        type_str = enum2name.get(jtype, "Unk")
        idx_q = int(m.jnt_qposadr[jid])
        idx_v = int(m.jnt_dofadr[jid])
        next_q = m.jnt_qposadr[jid + 1] if jid + 1 < m.njnt else m.nq
        next_v = m.jnt_dofadr[jid + 1] if jid + 1 < m.njnt else m.nv
        nq = int(next_q - idx_q)
        nv = int(next_v - idx_v)
        lines.append(f"{jid:3d} | {name:20s} | {type_str:6s} | {nq:2d} | {nv:2d} | {idx_q:5d} | {idx_v:4d}")

    lines.append("")
    trn_enum = mujoco.mjtTrn
    trn2name = {getattr(trn_enum, a): a[5:].title() for a in dir(trn_enum) if a.startswith("mjTRN_")}
    joint_names = {jid: m.names[m.name_jntadr[jid]:].split(b'\x00', 1)[0].decode() for jid in range(m.njnt)}
    lines.append(" id | name                 | trn     | target_joint")
    lines.append("----+----------------------+---------+-------------")
    for aid in range(m.nu):
        adr  = m.name_actuatoradr[aid]
        name = m.names[adr:].split(b'\x00', 1)[0].decode()
        trn_type = int(m.actuator_trntype[aid])
        trn_str  = trn2name.get(trn_type, "Unk")
        target_joint = "-"
        if trn_type in (trn_enum.mjTRN_JOINT, trn_enum.mjTRN_JOINTINPARENT):
            j_id = int(m.actuator_trnid[aid, 0])
            target_joint = joint_names.get(j_id, str(j_id))
        lines.append(f"{aid:3d} | {name:20s} | {trn_str:7s} | {target_joint}")

    lines.append("")
    sens_enum = mujoco.mjtSensor
    sens2name = {getattr(sens_enum, a): a[7:].title() for a in dir(sens_enum) if a.startswith("mjSENS_")}
    obj_enum  = mujoco.mjtObj
    obj2name  = {getattr(obj_enum, a): a[6:].title() for a in dir(obj_enum) if a.startswith("mjOBJ_")}
    body_names = {bid: m.names[m.name_bodyadr[bid]:].split(b'\0', 1)[0].decode() for bid in range(m.nbody)}
    site_names = {sid: m.names[m.name_siteadr[sid]:].split(b'\0', 1)[0].decode() for sid in range(m.nsite)}

    def obj_name(objtype, objid):
        if objtype == obj_enum.mjOBJ_BODY:
            return body_names.get(objid, str(objid))
        if objtype == obj_enum.mjOBJ_SITE:
            return site_names.get(objid, str(objid))
        if objtype == obj_enum.mjOBJ_JOINT:
            return joint_names.get(objid, str(objid))
        return str(objid)

    lines.append(" id | name                        | type             | dim | adr | target (obj)")
    lines.append("----+-----------------------------+------------------+-----+-----+----------------")
    for sid in range(m.nsensor):
        adr  = m.name_sensoradr[sid]
        name = m.names[adr:].split(b'\0', 1)[0].decode()
        stype = int(m.sensor_type[sid])
        tstr  = sens2name.get(stype, "Unk")
        dim   = int(m.sensor_dim[sid])
        sadr  = int(m.sensor_adr[sid])
        objtype = int(m.sensor_objtype[sid])
        objid   = int(m.sensor_objid[sid])
        target  = f"{obj2name.get(objtype,'-')}:{obj_name(objtype,objid)}" if objid >= 0 else "-"
        lines.append(f"{sid:3d} | {name:27s} | {tstr:16s} | {dim:3d} | {sadr:3d} | {target}")

    lines.append("")
    lines.append(" id | name                        | mode     | resolution")
    lines.append("----+-----------------------------+----------+------------")
    mode_map = {}
    cam_enum = getattr(mujoco, "mjtCamLight", None)
    if cam_enum is not None:
        for attr in dir(cam_enum):
            if attr.startswith("mjCAMLIGHT_"):
                mode_map[getattr(cam_enum, attr)] = attr[10:].title()
    else:
        mode_map = {0: "Fixed", 1: "Track", 2: "Trackcom"}

    try:
        offw = int(m.vis.global_.offwidth)
        offh = int(m.vis.global_.offheight)
        res_str = f"{offw}x{offh}" if offw > 0 and offh > 0 else "-"
    except Exception:
        res_str = "-"

    ncam = getattr(m, "ncam", 0)
    for cid in range(ncam):
        cadr  = m.name_camadr[cid]
        cname = m.names[cadr:].split(b'\x00', 1)[0].decode()
        if hasattr(m, "cam_mode"):
            try:
                mode_val = int(m.cam_mode[cid])
                mode_str = mode_map.get(mode_val, str(mode_val))
            except Exception:
                mode_str = "-"
        else:
            mode_str = "-"
        lines.append(f"{cid:3d} | {cname:27s} | {mode_str:8s} | {res_str}")
    lines.append("=================================================================")
    lines.append("=================================================================")
    return "\n".join(lines)

def to_NamedFloat64ArrayMsg(name: str, arr: np.ndarray) -> NamedFloat64Array:
    msg = NamedFloat64Array()
    msg.name = name
    f = Float64MultiArray()
    f.data = [float(x) for x in np.asarray(arr).ravel()]
    msg.value = f
    return msg
