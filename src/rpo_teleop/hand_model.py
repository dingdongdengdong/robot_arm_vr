"""AmazingHand — 서보 공간 제어 + 링크 pose 계산.

★ 핵심: 이 손은 8-DOF 가 아니라 "서보 8개 / 관절 12개" 다.

손가락마다 서보 2개가 평행 4절 + 차동 기구로 관절 3개(외전/근위굴곡/원위굴곡)를
움직인다. 즉 관절 12개가 서로 독립이 아니다.

    URDF 관절 리밋 사각형 중 실제로 도달 가능한 조합은 45.3% 뿐이다.
    (직접 검증: 14,400 격자점 중 6,529점 = 45.3%. 기구 담당자 문서와 일치)

그래서 **명령은 서보각 8개로 하고, 관절각 12개는 표시·기록용으로만 변환해서 쓴다.**
관절각을 직접 지령하면 실물에서 나올 수 없는 자세를 만들게 된다.

    트리거/손트래킹 ──(서보각 8개)──┬──> 실물 SCS0009 로 그대로 전송
                                  └──> servo_to_joint_flat() ──> 렌더링·데이터셋

보간도 반드시 서보 공간에서 한다. 관절 공간에서 보간하면 양 끝은 같아도 중간에서
최대 11.44° 어긋난다 (직접 검증, 기구 담당자 문서와 일치).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SERVO_MAP_DIR = Path(__file__).resolve().parent / "servo_map"
if str(_SERVO_MAP_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVO_MAP_DIR))

from hand_servo_map import (  # noqa: E402
    SERVO_LIMIT_RAD,
    is_reachable,
    joint_to_servo,
    servo_to_joint_flat,
)

HAND_URDF = Path(__file__).resolve().parents[2] / "assets" / "hand" / "hand.urdf"

# 실물 데모(AmazingHand_Demo.py)에서 가져온, 실기 검증된 자세. 단위 deg.
SERVO_OPEN_DEG = np.array([-35, 35, -35, 35, -35, 35, -35, 35], dtype=float)
SERVO_CLOSE_DEG = np.array([90, -90, 90, -90, 90, -90, 90, -90], dtype=float)

# 손가락 이름 (f4 가 엄지)
FINGER_NAMES = ["index", "middle", "ring", "thumb"]

# 팔 hand_mount → 손 hand_base 장착 변환 (기구 담당자 제공, 검산 완료).
# SuperArm HOME에서는 hand_mount 축과 world 축이 같으므로 사용자가 지정한
# world Y(+90°) 보정을 기본 RPY 앞에 곱한다. 최종 축은 손 +X=world +X,
# 손 +Y=world +Z, 손가락 진행축 +Z=world -Y 이다.
MOUNT_XYZ = np.array([0.0, 0.0, 0.0])
MOUNT_RPY = np.array([1.570796327, -1.570796327, 0.0])
MOUNT_WORLD_Y_ROT_RAD = np.pi / 2


def grasp_to_servo(t: float | np.ndarray) -> np.ndarray:
    """쥐는 정도 0.0(폄)~1.0(쥠) → 서보각 8개 (rad).

    ★ 반드시 서보 공간에서 보간한다. 관절 공간 보간은 중간에서 최대 11.44° 어긋난다.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    deg = SERVO_OPEN_DEG + (SERVO_CLOSE_DEG - SERVO_OPEN_DEG) * t
    return np.radians(deg)


def clamp_servo(servo_rad: np.ndarray) -> np.ndarray:
    """SCS0009 정격 ±90° 안으로."""
    return np.clip(np.asarray(servo_rad, dtype=float), -SERVO_LIMIT_RAD, SERVO_LIMIT_RAD)


class HandModel:
    """손 URDF FK + 서보/관절 변환.

    placo 대신 pinocchio 를 직접 쓴다. 이 URDF 는 관절과 링크가 이름을 공유해서
    (예: hand_f1_distal 이 관절이자 링크) placo 의 이름 기반 프레임 조회가
    "Several frames match the filter" 로 실패한다. pinocchio 는 FrameType.BODY 로
    링크만 명시적으로 고를 수 있다.
    """

    def __init__(self, urdf_path: str | Path = HAND_URDF):
        import pinocchio as pin

        self._pin = pin
        self.urdf_path = Path(urdf_path)
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        # URDF 순서 = 서보 매핑 출력 순서
        self.joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]
        self.link_names = [
            f.name for f in self.model.frames if f.type == pin.FrameType.BODY
        ]
        self._q = np.zeros(self.model.nq)
        self.set_joints(self._q)

    @property
    def n_joints(self) -> int:
        return int(self.model.nq)

    @property
    def n_servos(self) -> int:
        return 2 * len(FINGER_NAMES)

    def set_joints(self, q: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=float).copy()
        self._pin.forwardKinematics(self.model, self.data, self._q)
        self._pin.updateFramePlacements(self.model, self.data)

    def set_servos(self, servo_rad: np.ndarray) -> np.ndarray:
        """서보각 8개를 넣으면 관절각 12개로 변환해 자세를 갱신하고 그 값을 돌려준다."""
        q = servo_to_joint_flat(clamp_servo(servo_rad))
        self.set_joints(np.asarray(q, dtype=float))
        return self._q.copy()

    def set_grasp(self, t: float) -> np.ndarray:
        """쥐는 정도 0~1 로 자세를 잡는다."""
        return self.set_servos(grasp_to_servo(t))

    @property
    def joints(self) -> np.ndarray:
        return self._q.copy()

    def link_pose(self, name: str) -> np.ndarray:
        """링크의 4x4 pose (hand_base 기준)."""
        fid = self.model.getFrameId(name, self._pin.FrameType.BODY)
        return self.data.oMf[fid].homogeneous.copy()

    def link_poses(self) -> dict[str, np.ndarray]:
        return {n: self.link_pose(n) for n in self.link_names}

    # ── 손 트래킹 → 서보 (2단계 확장용) ───────────────────────────────
    @staticmethod
    def joints_to_servo(abduction: float, flexion: float) -> tuple[float, float]:
        """(외전, 굴곡) → 서보 2개. 도달 불가면 가장 가까운 값이 나온다."""
        return joint_to_servo(abduction, flexion)

    @staticmethod
    def reachable(abduction: float, flexion: float) -> bool:
        return bool(is_reachable(abduction, flexion))
