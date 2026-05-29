#!/usr/bin/env python3
"""Teleoperate robosuite / LIBERO simulation with a local Pika Sense device.

The script is self-contained inside this repository: it uses the vendored Pika
SDK under ``thiry_party/pika_sdk`` and does not depend on sibling projects.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import glob
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Iterable, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
PIKA_SDK_ROOT = REPO_ROOT / "thiry_party" / "pika_sdk"
LIBERO_ROOT = REPO_ROOT / "thiry_party" / "LIBERO"

if PIKA_SDK_ROOT.exists() and str(PIKA_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(PIKA_SDK_ROOT))
if LIBERO_ROOT.exists() and str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO_ROOT / "outputs" / ".libero"))

LOGGER = logging.getLogger("pika_sim_teleop")
logging.getLogger("pika.serial_comm").setLevel(logging.CRITICAL)


def ensure_local_libero_config() -> None:
    """Prevent vendored LIBERO from prompting or writing to ~/.libero."""
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = LIBERO_ROOT / "libero" / "libero"
    config = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": LIBERO_ROOT / "libero" / "datasets",
        "assets": benchmark_root / "assets",
    }
    lines = [f"{key}: {value}\n" for key, value in config.items()]
    config_file.write_text("".join(lines), encoding="utf-8")


class MathTools:
    """Small pose-conversion toolbox matching the Pika teleop convention."""

    def xyzrpy2mat(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        a, b = np.cos(yaw), np.sin(yaw)
        c, d = np.cos(pitch), np.sin(pitch)
        e, f = np.cos(roll), np.sin(roll)
        de, df = d * e, d * f
        transform[0, 0] = a * c
        transform[0, 1] = a * df - b * e
        transform[0, 2] = b * f + a * de
        transform[0, 3] = x
        transform[1, 0] = b * c
        transform[1, 1] = a * e + b * df
        transform[1, 2] = b * de - a * f
        transform[1, 3] = y
        transform[2, 0] = -d
        transform[2, 1] = c * f
        transform[2, 2] = c * e
        transform[2, 3] = z
        return transform

    def mat2xyzrpy(self, matrix: np.ndarray) -> list[float]:
        pitch_arg = float(np.clip(-matrix[2, 0], -1.0, 1.0))
        return [
            float(matrix[0, 3]),
            float(matrix[1, 3]),
            float(matrix[2, 3]),
            float(math.atan2(matrix[2, 1], matrix[2, 2])),
            float(math.asin(pitch_arg)),
            float(math.atan2(matrix[1, 0], matrix[0, 0])),
        ]

    def quaternion_to_rpy(self, x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(roll), float(pitch), float(yaw)

    def mat_to_quat(self, matrix: np.ndarray) -> np.ndarray:
        rot = np.asarray(matrix[:3, :3], dtype=np.float64)
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (rot[2, 1] - rot[1, 2]) / s
            y = (rot[0, 2] - rot[2, 0]) / s
            z = (rot[1, 0] - rot[0, 1]) / s
        else:
            idx = int(np.argmax(np.diag(rot)))
            if idx == 0:
                s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
                w = (rot[2, 1] - rot[1, 2]) / s
                x = 0.25 * s
                y = (rot[0, 1] + rot[1, 0]) / s
                z = (rot[0, 2] + rot[2, 0]) / s
            elif idx == 1:
                s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
                w = (rot[0, 2] - rot[2, 0]) / s
                x = (rot[0, 1] + rot[1, 0]) / s
                y = 0.25 * s
                z = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
                w = (rot[1, 0] - rot[0, 1]) / s
                x = (rot[0, 2] + rot[2, 0]) / s
                y = (rot[1, 2] + rot[2, 1]) / s
                z = 0.25 * s
        return self.quat_normalize(np.array([x, y, z, w], dtype=np.float64))

    def rpy_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return self.quat_normalize(
            np.array(
                [
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                    cr * cp * cy + sr * sp * sy,
                ],
                dtype=np.float64,
            )
        )

    @staticmethod
    def quat_normalize(quat: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return np.asarray(quat, dtype=np.float64) / norm

    def slerp(self, quat0: np.ndarray, quat1: np.ndarray, alpha: float) -> np.ndarray:
        q0 = self.quat_normalize(np.asarray(quat0, dtype=np.float64))
        q1 = self.quat_normalize(np.asarray(quat1, dtype=np.float64))
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return self.quat_normalize((1.0 - alpha) * q0 + alpha * q1)
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        a = math.sin((1.0 - alpha) * theta) / sin_theta
        b = math.sin(alpha * theta) / sin_theta
        return a * q0 + b * q1


def detect_pika_sense_port(preferred: str = "") -> str:
    if preferred:
        return preferred
    env_port = os.environ.get("PIKA_SENSE_PORT")
    if env_port:
        return env_port
    ports = sorted(glob.glob("/dev/ttyUSB*"))
    return ports[0] if ports else "/dev/ttyUSB0"


class PikaSense:
    """Local adapter around the vendored Pika SDK Sense device."""

    def __init__(
        self,
        port: str = "",
        tracker_device: str = "T20",
        tracker_config: Optional[str] = None,
        tracker_lh_config: Optional[str] = None,
    ):
        self.port = port
        self.tracker_device = tracker_device
        self.tracker_config = tracker_config
        self.tracker_lh_config = tracker_lh_config
        self._sense = None
        self._latest_pose: Optional[tuple[list[float], list[float]]] = None
        self._pose_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self) -> None:
        from pika.sense import Sense

        self.port = detect_pika_sense_port(self.port)
        self._sense = Sense(self.port)
        if not self._sense.connect():
            raise RuntimeError(f"failed to connect Pika Sense on {self.port}")
        if self.tracker_config or self.tracker_lh_config:
            self._sense.set_vive_tracker_config(
                config_path=self.tracker_config,
                lh_config=self.tracker_lh_config,
            )
        self._running = True
        self._thread = threading.Thread(target=self._tracker_loop, daemon=True)
        self._thread.start()
        print(f"[PikaSense] connected @ {self.port} (tracker={self.tracker_device})")

    def disconnect(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._sense is not None:
            try:
                self._sense.disconnect()
            except Exception:
                pass

    def _tracker_loop(self) -> None:
        while self._running:
            try:
                pose = self._sense.get_pose(self.tracker_device)
            except Exception as exc:
                LOGGER.debug("Pika tracker poll failed: %s", exc)
                pose = None
            if pose is not None:
                with self._pose_lock:
                    self._latest_pose = (list(pose.position), list(pose.rotation))
            time.sleep(0.02)

    def get_tracker_pose(self) -> Optional[tuple[list[float], list[float]]]:
        with self._pose_lock:
            return self._latest_pose

    def wait_for_tracker(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.get_tracker_pose() is not None:
                return True
            time.sleep(0.05)
        return False

    def get_encoder_rad(self) -> float:
        if self._sense is None:
            return 0.0
        try:
            return float(self._sense.get_encoder_data().get("rad", 0.0))
        except Exception:
            return 0.0

    def get_command_state(self) -> int:
        if self._sense is None:
            return 0
        try:
            return int(self._sense.get_command_state())
        except Exception:
            return 0

    def is_alive(self) -> bool:
        if self._sense is None:
            return False
        serial_comm = getattr(self._sense, "serial_comm", None)
        if serial_comm is None or not getattr(serial_comm, "is_connected", False):
            return False
        serial = getattr(serial_comm, "serial", None)
        if serial is None:
            return False
        try:
            _ = serial.in_waiting
            return True
        except Exception:
            return False


class DummyPikaSense:
    """Dry-run device: no tracker motion, open gripper, never engaged."""

    def connect(self) -> None:
        print("[PikaSense] dry-run dummy active")

    def disconnect(self) -> None:
        return None

    def get_tracker_pose(self):
        return None

    def wait_for_tracker(self, timeout: float = 0.0) -> bool:
        del timeout
        return False

    def get_encoder_rad(self) -> float:
        return 1.7

    def get_command_state(self) -> int:
        return 0

    def is_alive(self) -> bool:
        return True


def rotation_matrix_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    if sy > 1e-6:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def complete_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size == action_dim:
        return action
    if action.size < action_dim:
        return np.concatenate([action, np.zeros(action_dim - action.size, dtype=np.float32)])
    return action[:action_dim]


class PikaSimController:
    """Map Pika tracker deltas into normalized simulation controller actions."""

    def __init__(
        self,
        sense,
        *,
        control_mode: str,
        position_scale: float,
        action_pos_gain: float,
        action_rot_gain: float,
        pos_deadband_m: float,
        rot_deadband_rad: float,
        pose_alpha: float,
        gripper_alpha: float,
        pika_to_arm: Iterable[float],
        delta_pos_map: Iterable[float],
        delta_rot_map: Iterable[float],
        pika_closed_rad: float,
        pika_open_rad: float,
    ):
        self.sense = sense
        self.tools = MathTools()
        self.control_mode = control_mode
        if self.control_mode not in ("target", "delta"):
            raise ValueError("control_mode must be 'target' or 'delta'")
        self.position_scale = float(position_scale)
        self.action_pos_gain = float(action_pos_gain)
        self.action_rot_gain = float(action_rot_gain)
        self.pos_deadband_m = max(0.0, float(pos_deadband_m))
        self.rot_deadband_rad = max(0.0, float(rot_deadband_rad))
        self.pose_alpha = float(np.clip(pose_alpha, 0.0, 1.0))
        self.gripper_alpha = float(np.clip(gripper_alpha, 0.0, 1.0))
        self.pika_to_arm = list(float(v) for v in pika_to_arm)
        self.delta_pos_map = np.asarray(list(float(v) for v in delta_pos_map), dtype=np.float64).reshape(3, 3)
        self.delta_rot_map = np.asarray(list(float(v) for v in delta_rot_map), dtype=np.float64).reshape(3, 3)
        self.pika_closed_rad = float(pika_closed_rad)
        self.pika_open_rad = float(pika_open_rad)
        self._smoothed_pos: Optional[np.ndarray] = None
        self._smoothed_quat: Optional[np.ndarray] = None
        self._smoothed_gripper: Optional[float] = None
        self._tracker_xyzrpy: Optional[list[float]] = None
        self._tracker_base: Optional[list[float]] = None
        self._tcp_base: Optional[list[float]] = None
        self._last_delta_tracker: Optional[list[float]] = None
        self._active = False
        self._last_trigger: Optional[int] = None

    @property
    def active(self) -> bool:
        return self._active

    def _adjust_pika_to_arm(self, xyzrpy: list[float]) -> list[float]:
        transform = self.tools.xyzrpy2mat(*xyzrpy)
        adjust = self.tools.xyzrpy2mat(*self.pika_to_arm)
        return self.tools.mat2xyzrpy(transform @ adjust)

    def _refresh_tracker_pose(self) -> None:
        pose = self.sense.get_tracker_pose()
        if pose is None:
            return
        position, quat = pose
        raw_pos = np.asarray(position, dtype=np.float64)
        raw_quat = self.tools.quat_normalize(np.asarray(quat, dtype=np.float64))
        if self._smoothed_pos is not None and self.pose_alpha < 1.0:
            self._smoothed_pos = self.pose_alpha * raw_pos + (1.0 - self.pose_alpha) * self._smoothed_pos
            self._smoothed_quat = self.tools.slerp(self._smoothed_quat, raw_quat, self.pose_alpha)
        else:
            self._smoothed_pos = raw_pos
            self._smoothed_quat = raw_quat
        roll, pitch, yaw = self.tools.quaternion_to_rpy(*self._smoothed_quat)
        xyzrpy = [float(self._smoothed_pos[0]), float(self._smoothed_pos[1]), float(self._smoothed_pos[2]), roll, pitch, yaw]
        self._tracker_xyzrpy = self._adjust_pika_to_arm(xyzrpy)

    def _handle_trigger(self, current_tcp_xyzrpy: list[float]) -> None:
        current = self.sense.get_command_state()
        if self._last_trigger is None:
            self._last_trigger = current
            return
        if current == self._last_trigger:
            return
        self._last_trigger = current
        self._active = not self._active
        if self._active:
            if self._tracker_xyzrpy is not None:
                self._tracker_base = list(self._tracker_xyzrpy)
                self._tcp_base = list(current_tcp_xyzrpy)
                self._last_delta_tracker = list(self._tracker_xyzrpy)
            print("[PikaSim] >> ENGAGED")
        else:
            self._tracker_base = None
            self._tcp_base = None
            self._last_delta_tracker = None
            print("[PikaSim] << RELEASED")

    def _target_tcp(self) -> Optional[list[float]]:
        if self._tracker_xyzrpy is None or self._tracker_base is None or self._tcp_base is None:
            return None
        begin = self.tools.xyzrpy2mat(*self._tracker_base)
        zero = self.tools.xyzrpy2mat(*self._tcp_base)
        end = self.tools.xyzrpy2mat(*self._tracker_xyzrpy)
        delta = np.linalg.inv(begin) @ end
        delta[:3, :3] = self.delta_rot_map @ delta[:3, :3] @ self.delta_rot_map.T
        delta[:3, 3] = self.delta_pos_map @ delta[:3, 3]
        delta[:3, 3] *= self.position_scale
        return self.tools.mat2xyzrpy(zero @ delta)

    def _zero_action(self, grip: float, action_dim: int) -> np.ndarray:
        return complete_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip], dtype=np.float32), action_dim)

    def _mapped_delta_to_action(self, delta: np.ndarray, grip: float, action_dim: int) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float64).copy()
        delta[:3, :3] = self.delta_rot_map @ delta[:3, :3] @ self.delta_rot_map.T
        delta[:3, 3] = self.delta_pos_map @ delta[:3, 3]
        delta_pos = delta[:3, 3] * self.position_scale
        delta_rpy = np.asarray(rotation_matrix_to_rpy(delta[:3, :3]), dtype=np.float64)
        if float(np.linalg.norm(delta_pos)) < self.pos_deadband_m:
            delta_pos[:] = 0.0
        if float(np.linalg.norm(delta_rpy)) < self.rot_deadband_rad:
            delta_rpy[:] = 0.0
        delta_pos = delta_pos * self.action_pos_gain
        delta_rpy = delta_rpy * self.action_rot_gain
        full_action = np.concatenate([delta_pos, delta_rpy, np.asarray([grip], dtype=np.float64)])
        return complete_action(np.clip(full_action, -1.0, 1.0).astype(np.float32), action_dim)

    def _delta_action(self, grip: float, action_dim: int) -> np.ndarray:
        if self._tracker_xyzrpy is None:
            return self._zero_action(grip, action_dim)
        if self._last_delta_tracker is None:
            self._last_delta_tracker = list(self._tracker_xyzrpy)
            return self._zero_action(grip, action_dim)
        previous = self.tools.xyzrpy2mat(*self._last_delta_tracker)
        current = self.tools.xyzrpy2mat(*self._tracker_xyzrpy)
        self._last_delta_tracker = list(self._tracker_xyzrpy)
        delta = np.linalg.inv(previous) @ current
        return self._mapped_delta_to_action(delta, grip, action_dim)

    def gripper_action(self) -> float:
        raw = self.sense.get_encoder_rad()
        if self._smoothed_gripper is not None and self.gripper_alpha < 1.0:
            raw = self.gripper_alpha * raw + (1.0 - self.gripper_alpha) * self._smoothed_gripper
        self._smoothed_gripper = float(raw)
        denom = self.pika_open_rad - self.pika_closed_rad
        t = 0.0 if abs(denom) < 1e-9 else (float(raw) - self.pika_closed_rad) / denom
        t = float(np.clip(t, 0.0, 1.0))
        return float(1.0 + t * (-2.0))

    def action(self, current_tcp_xyzrpy: list[float], action_dim: int) -> np.ndarray:
        self._refresh_tracker_pose()
        self._handle_trigger(current_tcp_xyzrpy)
        grip = self.gripper_action()
        if not self._active:
            self._last_delta_tracker = None
            return self._zero_action(grip, action_dim)
        if self.control_mode == "delta":
            return self._delta_action(grip, action_dim)
        target = self._target_tcp()
        if target is None:
            return self._zero_action(grip, action_dim)

        current = np.asarray(current_tcp_xyzrpy, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        delta_pos = target[:3] - current[:3]
        delta_rpy = (target[3:6] - current[3:6] + np.pi) % (2.0 * np.pi) - np.pi
        if float(np.linalg.norm(delta_pos)) < self.pos_deadband_m:
            delta_pos[:] = 0.0
        if float(np.linalg.norm(delta_rpy)) < self.rot_deadband_rad:
            delta_rpy[:] = 0.0
        delta_pos = delta_pos * self.action_pos_gain
        delta_rpy = delta_rpy * self.action_rot_gain
        full_action = np.concatenate([delta_pos, delta_rpy, np.asarray([grip], dtype=np.float64)])
        return complete_action(np.clip(full_action, -1.0, 1.0).astype(np.float32), action_dim)


class SimBackend:
    name = "base"

    def reset(self):
        raise NotImplementedError

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def task_description(self) -> str:
        return ""

    def current_tcp_xyzrpy(self) -> list[float]:
        raise NotImplementedError

    def state_vector(self) -> np.ndarray:
        raise NotImplementedError

    def images(self, cameras: Iterable[str], width: int, height: int) -> dict[str, np.ndarray]:
        return {}

    def check_success(self) -> bool:
        return False

    def check_failure(self) -> tuple[bool, str]:
        return False, ""


class RobosuiteLikeMixin:
    env = None

    @property
    def action_dim(self) -> int:
        return int(getattr(self.env, "action_dim", getattr(getattr(self.env, "env", None), "action_dim", 7)))

    def _robot(self):
        return self.env.robots[0]

    def current_tcp_xyzrpy(self) -> list[float]:
        sim = self.env.sim
        for body_name in ("gripper0_eef", "robot0_eef"):
            try:
                body_id = sim.model.body_name2id(body_name)
                pos = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
                rot = np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
                return [float(pos[0]), float(pos[1]), float(pos[2]), *rotation_matrix_to_rpy(rot)]
            except Exception:
                pass
        for site_name in ("gripper0_grip_site", "robot0_grip_site"):
            try:
                site_id = sim.model.site_name2id(site_name)
                pos = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
                rot = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
                return [float(pos[0]), float(pos[1]), float(pos[2]), *rotation_matrix_to_rpy(rot)]
            except Exception:
                pass
        raise RuntimeError("could not locate gripper TCP body/site in simulation")

    def state_vector(self) -> np.ndarray:
        robot = self._robot()
        arm_indexes = getattr(robot, "_ref_joint_pos_indexes", [])[:6]
        arm = np.asarray(self.env.sim.data.qpos[arm_indexes], dtype=np.float32)
        gripper_indexes = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
        if gripper_indexes:
            grip = float(np.mean(np.asarray(self.env.sim.data.qpos[gripper_indexes], dtype=np.float32)))
        else:
            grip = 0.0
        return np.concatenate([arm, np.asarray([grip], dtype=np.float32)])

    def images(self, cameras: Iterable[str], width: int, height: int) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for camera in cameras:
            rgb = self.env.sim.render(camera_name=camera, width=width, height=height)
            out[camera] = np.asarray(rgb[::-1], dtype=np.uint8)
        return out


class UprightBlocksBackend(RobosuiteLikeMixin, SimBackend):
    name = "upright_blocks"

    def __init__(self, args: argparse.Namespace):
        from robosuite.controllers import load_controller_config
        from robosuite.wrappers import VisualizationWrapper
        from create_robosuite_upright_blocks_scene import UprightBlocksLift

        controller_config = load_controller_config(default_controller=args.controller)
        base_env = UprightBlocksLift(
            controller_configs=controller_config,
            has_renderer=not args.dry_run,
            has_offscreen_renderer=True,
            render_camera=args.viewer_camera,
            camera_names=list(args.cameras),
            camera_widths=args.image_width,
            camera_heights=args.image_height,
            use_camera_obs=False,
            ignore_done=True,
            horizon=args.max_steps if args.max_steps > 0 else 1000,
            control_freq=args.control_freq,
            hard_reset=False,
        )
        self.env = base_env if args.dry_run else VisualizationWrapper(base_env)
        self._task = "pick red cube and place it on the plate without tipping the yellow slabs"

    @property
    def task_description(self) -> str:
        return self._task

    def reset(self):
        return self.env.reset()

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    def check_success(self) -> bool:
        return bool(self.env._check_success())

    def check_failure(self) -> tuple[bool, str]:
        if self.env._check_obstacle_violation():
            return True, "yellow slab tipped"
        return False, ""


class LiberoBackend(RobosuiteLikeMixin, SimBackend):
    name = "libero"

    def __init__(self, args: argparse.Namespace):
        ensure_local_libero_config()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bddl_file = args.bddl_file
        self._task = ""
        if bddl_file is None:
            benchmark_dict = benchmark.get_benchmark_dict()
            if args.benchmark not in benchmark_dict:
                raise ValueError(f"unknown LIBERO benchmark {args.benchmark!r}; choices={sorted(benchmark_dict)}")
            suite = benchmark_dict[args.benchmark]()
            task = suite.get_task(args.task_id)
            bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            self._task = str(task.language)
        else:
            bddl_file = Path(bddl_file)
            self._task = bddl_file.stem

        self._bddl_file = Path(bddl_file)
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(self._bddl_file),
            robots=[args.libero_robot],
            controller=args.controller,
            gripper_types="default",
            has_renderer=not args.dry_run,
            has_offscreen_renderer=True,
            render_camera=args.viewer_camera,
            camera_names=list(args.cameras),
            camera_heights=args.image_height,
            camera_widths=args.image_width,
            use_camera_obs=False,
            control_freq=args.control_freq,
            horizon=args.max_steps if args.max_steps > 0 else 1000,
            ignore_done=True,
        )

    @property
    def task_description(self) -> str:
        return self._task

    def reset(self):
        return self.env.reset()

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def render(self) -> None:
        if getattr(self.env.env, "has_renderer", False):
            self.env.env.render()

    def close(self) -> None:
        self.env.env.close()

    def check_success(self) -> bool:
        return bool(self.env.check_success())


def make_backend(args: argparse.Namespace) -> SimBackend:
    try:
        if args.backend == "upright_blocks":
            return UprightBlocksBackend(args)
        if args.backend == "libero":
            return LiberoBackend(args)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"missing dependency for backend {args.backend!r}: {exc.name}. "
            "Run this in the robosuite/LIBERO simulation environment."
        ) from exc
    raise ValueError(f"unsupported backend: {args.backend}")


def save_episode(
    output_dir: Path,
    frames: list[dict],
    metadata: dict,
    cameras: Iterable[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"episode_{timestamp}.npz"
    payload: dict[str, object] = {
        "states": np.asarray([frame["state"] for frame in frames], dtype=np.float32),
        "actions": np.asarray([frame["action"] for frame in frames], dtype=np.float32),
        "controller_actions": np.asarray([frame["controller_action"] for frame in frames], dtype=np.float32),
        "metadata": json.dumps(metadata, ensure_ascii=False, indent=2),
    }
    for camera in cameras:
        imgs = [frame["images"][camera] for frame in frames if camera in frame["images"]]
        if imgs:
            payload[f"images_{camera}"] = np.asarray(imgs, dtype=np.uint8)
    np.savez_compressed(path, **payload)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["upright_blocks", "libero"], default="upright_blocks")
    parser.add_argument("--controller", choices=["OSC_POSE", "IK_POSE"], default="OSC_POSE")
    parser.add_argument("--dry-run", action="store_true", help="Use dummy Pika input and avoid interactive hardware.")
    parser.add_argument("--record", action="store_true", help="Save a raw npz episode.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "pika_sim_teleop")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until success/failure/operator interrupt.")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument(
        "--control-mode",
        choices=["target", "delta"],
        default="target",
        help="target tracks an accumulated TCP target; delta sends per-frame Pika deltas directly as sim actions.",
    )
    parser.add_argument("--success-hold", type=int, default=10)
    parser.add_argument("--viewer-camera", default="frontview")
    parser.add_argument("--cameras", nargs="+", default=["frontview"])
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)

    parser.add_argument("--sense-port", default="", help="Pika Sense serial port; empty means auto-detect.")
    parser.add_argument("--tracker-device", default="T20")
    parser.add_argument("--tracker-config", default=None)
    parser.add_argument("--tracker-lh-config", default=None)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--action-pos-gain", type=float, default=20.0)
    parser.add_argument("--action-rot-gain", type=float, default=2.0)
    parser.add_argument("--pos-deadband-m", type=float, default=0.002, help="Ignore TCP position error smaller than this many meters.")
    parser.add_argument("--rot-deadband-deg", type=float, default=3.0, help="Ignore TCP rotation error smaller than this many degrees.")
    parser.add_argument("--pose-alpha", type=float, default=0.3)
    parser.add_argument("--gripper-alpha", type=float, default=0.9)
    parser.add_argument("--pika-to-arm", type=float, nargs=6, default=[0.0, 0.0, 0.0, 1.703151, 1.539109, 1.728148])
    parser.add_argument(
        "--delta-pos-map",
        type=float,
        nargs=9,
        default=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metavar=("M00", "M01", "M02", "M10", "M11", "M12", "M20", "M21", "M22"),
        help="Row-major 3x3 map from Pika translation delta to sim TCP translation delta.",
    )
    parser.add_argument(
        "--delta-rot-map",
        type=float,
        nargs=9,
        default=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="Row-major 3x3 axis map for Pika relative rotation.",
    )
    parser.add_argument("--pika-closed-rad", type=float, default=0.0)
    parser.add_argument("--pika-open-rad", type=float, default=1.7)

    parser.add_argument("--benchmark", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--bddl-file", type=Path, default=None)
    parser.add_argument("--libero-robot", default="Panda")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Optional[Path]:
    backend = make_backend(args)
    sense = DummyPikaSense() if args.dry_run else PikaSense(
        port=args.sense_port,
        tracker_device=args.tracker_device,
        tracker_config=args.tracker_config,
        tracker_lh_config=args.tracker_lh_config,
    )
    controller = PikaSimController(
        sense,
        control_mode=args.control_mode,
        position_scale=args.position_scale,
        action_pos_gain=args.action_pos_gain,
        action_rot_gain=args.action_rot_gain,
        pos_deadband_m=args.pos_deadband_m,
        rot_deadband_rad=math.radians(args.rot_deadband_deg),
        pose_alpha=args.pose_alpha,
        gripper_alpha=args.gripper_alpha,
        pika_to_arm=args.pika_to_arm,
        delta_pos_map=args.delta_pos_map,
        delta_rot_map=args.delta_rot_map,
        pika_closed_rad=args.pika_closed_rad,
        pika_open_rad=args.pika_open_rad,
    )
    frames: list[dict] = []
    success_hold = -1
    saved_path: Optional[Path] = None
    reason = "interrupted"
    success = False

    try:
        sense.connect()
        if not args.dry_run and not sense.wait_for_tracker(timeout=30.0):
            print("[warn] tracker pose not received in 30s; arm will move after tracker locks")
        backend.reset()
        if not args.dry_run:
            backend.render()

        step = 0
        dt = 1.0 / float(args.control_freq)
        last_health = 0.0
        while args.max_steps == 0 or step < args.max_steps:
            t0 = time.time()
            if t0 - last_health > 0.5:
                last_health = t0
                if not sense.is_alive():
                    reason = "Pika Sense USB serial dropped"
                    print(f"[error] {reason}; check USB cable, hub power, or udev reconnects.")
                    break
            state = backend.state_vector().astype(np.float32)
            images = backend.images(args.cameras, args.image_width, args.image_height) if args.record else {}
            sim_action = controller.action(backend.current_tcp_xyzrpy(), backend.action_dim)
            backend.step(sim_action)
            if not args.dry_run:
                backend.render()
            next_state = backend.state_vector().astype(np.float32)
            if args.record:
                frames.append(
                    {
                        "state": state,
                        "action": (next_state - state).astype(np.float32),
                        "controller_action": sim_action.astype(np.float32),
                        "images": images,
                    }
                )

            failed, failure_reason = backend.check_failure()
            if failed:
                reason = failure_reason
                break
            if backend.check_success():
                if success_hold == 0:
                    success = True
                    reason = "success"
                    break
                success_hold = args.success_hold if success_hold < 0 else success_hold - 1
            else:
                success_hold = -1

            step += 1
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
        else:
            reason = "max steps reached"

    except KeyboardInterrupt:
        reason = "operator interrupt"
    finally:
        if args.record:
            metadata = {
                "backend": args.backend,
                "task": backend.task_description,
                "controller": args.controller,
                "control_mode": args.control_mode,
                "action_dim": backend.action_dim,
                "success": success,
                "reason": reason,
                "control_freq": args.control_freq,
                "pika_to_arm": list(args.pika_to_arm),
                "delta_pos_map": list(args.delta_pos_map),
                "delta_rot_map": list(args.delta_rot_map),
                "position_scale": args.position_scale,
                "pika_closed_rad": args.pika_closed_rad,
                "pika_open_rad": args.pika_open_rad,
            }
            saved_path = save_episode(args.output_dir / args.backend, frames, metadata, args.cameras)
            print(f"[record] saved {len(frames)} frames to {saved_path}")
        try:
            backend.close()
        finally:
            sense.disconnect()
    print(f"[done] success={success} reason={reason}")
    return saved_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    try:
        run(args)
    except RuntimeError as exc:
        raise SystemExit(f"[error] {exc}") from None


if __name__ == "__main__":
    main()
