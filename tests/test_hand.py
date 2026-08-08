"""AmazingHand 서보 매핑 / 장착 변환 검증.

기구 담당자 문서(handoff/05_HAND_URDF_RESPONSE.txt)의 주장을 그대로 믿지 않고
여기서 수치로 확인한다. 특히 아래 둘은 틀리면 실물이 이상하게 움직인다.
  · 관절 리밋의 55%가 도달 불가 → 관절각을 직접 지령하면 안 된다
  · 관절 공간 보간은 서보 공간 보간과 최대 11.44° 어긋난다
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpo_teleop.hand_model import (  # noqa: E402
    MOUNT_RPY,
    MOUNT_WORLD_Y_ROT_RAD,
    SERVO_CLOSE_DEG,
    SERVO_OPEN_DEG,
    HandModel,
    clamp_servo,
    grasp_to_servo,
)

pytestmark = pytest.mark.skipif(
    not (ROOT / "assets" / "hand" / "hand.urdf").exists(),
    reason="손 URDF 가 아직 없음",
)


@pytest.fixture(scope="module")
def hand():
    return HandModel()


# ── 구조 ──────────────────────────────────────────────────────────────
def test_servo_and_joint_counts(hand):
    """서보 8개 / 관절 12개. 이걸 8-DOF 로 착각하면 제어가 통째로 틀어진다."""
    assert hand.n_servos == 8
    assert hand.n_joints == 12
    assert len(hand.link_names) == 13   # hand_base + 손가락 4 × 3


def test_total_mass_matches_spec():
    """손 질량 0.400 kg — 팔 부하 계산의 입력값이라 틀리면 안 된다."""
    import xml.etree.ElementTree as ET

    root = ET.parse(ROOT / "assets" / "hand" / "hand.urdf").getroot()
    total = sum(float(m.get("value"))
                for ln in root.findall("link")
                if (m := ln.find("inertial/mass")) is not None)
    assert np.isclose(total, 0.400, atol=1e-4)


# ── 서보 프리셋 (실물 데모에서 가져온 값) ──────────────────────────────
@pytest.mark.parametrize(
    "servo_deg, expect_ABC",
    [
        ([-35, 35], [-0.07, -28.85, -28.78]),   # 편 상태
        ([90, -90], [-0.37, +44.33, +37.84]),   # 쥔 상태
        ([0, 0], [0.00, 0.00, -0.01]),          # 모델 영점
        ([50, -50], [-0.13, +31.79, +29.24]),   # 중간
    ],
)
def test_servo_presets(hand, servo_deg, expect_ABC):
    servo8 = np.radians(list(servo_deg) * 4)
    q = hand.set_servos(servo8)
    got = np.degrees(q[:3])
    assert np.allclose(got, expect_ABC, atol=0.02), f"{got} != {expect_ABC}"


def test_grasp_endpoints_match_presets(hand):
    assert np.allclose(np.degrees(grasp_to_servo(0.0)), SERVO_OPEN_DEG)
    assert np.allclose(np.degrees(grasp_to_servo(1.0)), SERVO_CLOSE_DEG)


def test_grasp_clamped_outside_0_1():
    assert np.allclose(grasp_to_servo(-5.0), grasp_to_servo(0.0))
    assert np.allclose(grasp_to_servo(99.0), grasp_to_servo(1.0))


def test_servo_clamped_to_rating():
    """SCS0009 정격 ±90°. 넘겨 보내면 서보가 못 따라가거나 손상된다."""
    out = clamp_servo(np.radians([200, -200, 0, 0, 0, 0, 0, 0]))
    assert np.isclose(np.degrees(out[0]), 90.0)
    assert np.isclose(np.degrees(out[1]), -90.0)


# ── ★ 서보 공간 보간이 필수인 이유 ────────────────────────────────────
def test_joint_space_interpolation_differs_from_servo_space(hand):
    """관절 공간에서 보간하면 실물과 달라진다.

    양 끝은 같지만 중간에서 크게 어긋난다. 이 테스트가 깨지면 누군가
    grasp 보간을 관절 공간으로 바꿨다는 뜻이다.
    """
    q_open = hand.set_servos(grasp_to_servo(0.0)).copy()
    q_close = hand.set_servos(grasp_to_servo(1.0)).copy()

    worst = 0.0
    for t in np.linspace(0, 1, 21):
        q_servo = hand.set_servos(grasp_to_servo(t))          # 올바른 방식
        q_joint = q_open + (q_close - q_open) * t             # 잘못된 방식
        worst = max(worst, float(np.max(np.abs(np.degrees(q_servo - q_joint)))))

    assert worst > 8.0, f"차이가 {worst:.2f}° 뿐 — 매핑이 선형이 되어버렸는지 확인"
    assert worst < 15.0, f"차이가 {worst:.2f}° 로 예상(약 11.4°)보다 큼 — 매핑 변경 의심"


def test_grasp_is_monotonic(hand):
    """쥐는 정도를 올리면 굴곡도 단조 증가해야 한다 (조작 감각의 기본)."""
    flex = [float(hand.set_servos(grasp_to_servo(t))[1]) for t in np.linspace(0, 1, 21)]
    assert all(b >= a - 1e-9 for a, b in zip(flex, flex[1:])), f"단조 아님: {np.degrees(flex)}"


# ── ★ 관절 리밋을 조작 범위로 쓰면 안 되는 이유 ───────────────────────
def test_most_of_joint_limit_box_is_unreachable():
    """URDF 리밋 사각형의 절반 이상이 실물에서 나올 수 없는 조합이다."""
    A = np.linspace(-38.6868, 38.2, 60)
    B = np.linspace(-61.7, 44.2, 60)
    ok = np.array([[HandModel.reachable(np.radians(a), np.radians(b)) for a in A] for b in B])
    frac = float(ok.mean())
    assert 0.40 < frac < 0.50, f"도달 가능 비율 {frac:.1%} (문서 45.3%)"


def test_inverse_mapping_roundtrip():
    """(외전, 굴곡) → 서보 → 관절 왕복이 일치해야 손 트래킹 확장에 쓸 수 있다."""
    from hand_servo_map import servo_to_joint

    rng = np.random.default_rng(0)
    for _ in range(100):
        t1, t2 = rng.uniform(-np.pi / 2, np.pi / 2, 2)
        A, B, _ = servo_to_joint(t1, t2)
        r1, r2 = HandModel.joints_to_servo(A, B)
        A2, B2, _ = servo_to_joint(r1, r2)
        assert abs(np.degrees(A2 - A)) < 0.01
        assert abs(np.degrees(B2 - B)) < 0.01


# ── 장착 변환 ─────────────────────────────────────────────────────────
def test_mount_transform_orientation(hand):
    """SuperArm HOME에서 현재 손을 world Y축으로 +90도 회전한다.

    이게 틀리면 손이 90° 돌아간 채 달린다. 기구 담당자 문서의 핵심 검산값.
    """
    import placo

    from rpo_teleop.arm_visual import rpy_to_matrix

    arm = placo.RobotWrapper(
        str(ROOT / "assets" / "superarm_j1_j2" / "superarm_j1_j2.urdf")
    )
    arm.update_kinematics()
    T_mount = arm.get_T_world_frame("hand_mount")
    R = (
        T_mount[:3, :3]
        @ rpy_to_matrix((0.0, MOUNT_WORLD_Y_ROT_RAD, 0.0))
        @ rpy_to_matrix(MOUNT_RPY)
    )

    assert np.allclose(R[:, 0], [1, 0, 0], atol=1e-6), f"손바닥 +X → {R[:, 0]}"
    assert np.allclose(R[:, 1], [0, 0, 1], atol=1e-6), f"손 +Y → {R[:, 1]}"
    assert np.allclose(R[:, 2], [0, -1, 0], atol=1e-6), f"손가락 +Z → {R[:, 2]}"
    assert np.isclose(MOUNT_WORLD_Y_ROT_RAD, np.pi / 2)


def test_mount_height_checksum(hand):
    """관절각 0 에서 hand_base 높이 533.3 mm (기구 담당자 검산값)."""
    import placo

    arm = placo.RobotWrapper(str(ROOT / "assets" / "robot_arm" / "robot_arm.urdf"))
    arm.update_kinematics()
    z = arm.get_T_world_frame("hand_mount")[2, 3] * 1000
    assert np.isclose(z, 533.3, atol=0.2), f"{z:.1f} mm"
