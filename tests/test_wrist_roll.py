"""손목 롤 축 검증.

★ 축을 코드에 박으면 안 된다. 팔마다 다르고 **부호까지 반대**다.

    config/arm.json       joint5  EE로컬 축 = [0, -1, 0]
    config/arm_temp.json  joint3  EE로컬 축 = [0, +1, 0]

예전 구현은 EE 로컬 X축을 썼는데 실제 축은 둘 다 Y 였다. 그래서 팔이 낼 수
없는 회전을 요청했고, 자세가 soft task 라 예외 없이 엉뚱한 자세로 수렴했다.
전체 팔은 자세오차 117.7° 에 음의 방향이 죽었고, 임시 팔은 끝관절이 아예
안 움직였다. (Isaac 담당이 사용자 제보로 발견 — board/20_ISAAC.txt M-9)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpo_teleop.arm_config import ArmConfig  # noqa: E402
from rpo_teleop.transforms import rotvec_to_rotation  # noqa: E402

_spec = importlib.util.spec_from_file_location("sim", ROOT / "scripts" / "05_teleop_sim.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

CONFIGS = [ROOT / "config" / "arm.json", ROOT / "config" / "arm_temp.json"]


def test_b_button_software_stop_is_rising_edge_and_latched() -> None:
    latched = previous = False
    latched, previous, newly = sim.latch_software_stop(latched, True, previous)
    assert latched and previous and newly

    latched, previous, newly = sim.latch_software_stop(latched, False, previous)
    assert latched and not previous and not newly

    latched, previous, newly = sim.latch_software_stop(latched, True, previous)
    assert latched and previous and not newly


def test_b_button_software_stop_holds_without_disabling() -> None:
    class Motors:
        def __init__(self) -> None:
            self.hold_calls = 0
            self.disable_calls = 0
            self.writes = []

        def read_positions(self):
            return np.array([0.2, -0.3])

        def hold(self):
            self.hold_calls += 1

        def disable(self):
            self.disable_calls += 1

        def write_positions(self, q, **kwargs):
            self.writes.append((np.asarray(q), kwargs))

    motors = Motors()
    actual = sim.hold_current_position(motors, 2)

    assert np.allclose(actual, [0.2, -0.3])
    assert motors.hold_calls == 1
    assert motors.disable_calls == 0
    assert np.allclose(motors.writes[0][0], actual)
    assert np.allclose(motors.writes[0][1]["dq"], np.zeros(2))
    assert motors.writes[0][1]["engaged"] is False


@pytest.fixture(params=CONFIGS, ids=lambda p: p.stem)
def arm(request):
    if not request.param.exists():
        pytest.skip(f"{request.param.name} 없음")
    cfg = ArmConfig.load(request.param)
    return cfg, sim.ArmIK(cfg)


def test_measured_axis_matches_actual_joint_motion(arm):
    """실측한 축이 실제로 끝관절이 EE 를 돌리는 축과 같아야 한다."""
    cfg, ik = arm
    axis, lo, hi = sim.measure_wrist_axis(ik, cfg)
    assert np.isclose(np.linalg.norm(axis), 1.0), "단위벡터가 아니다"

    # 끝관절을 크게 돌려서 EE 가 정말 그 축으로 도는지 본다
    q0 = np.asarray(cfg.home, dtype=float)
    ik.set_q(q0); T0 = ik.fk()
    q1 = q0.copy(); q1[-1] += np.radians(30.0)
    ik.set_q(q1); T1 = ik.fk()
    R = T0[:3, :3].T @ T1[:3, :3]
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    got = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(ang))
    got /= np.linalg.norm(got)
    # ★ 부호까지 봐야 한다. abs 로 비교하면 반대 방향을 놓친다.
    assert np.allclose(got, axis, atol=1e-3), f"축이 다르다: 실측 {axis} vs 실제 {got}"


def test_roll_is_not_the_local_x_axis(arm):
    """★ 예전 버그: EE 로컬 X 축을 썼다. 이 팔들은 둘 다 Y 축이다."""
    cfg, ik = arm
    axis, _, _ = sim.measure_wrist_axis(ik, cfg)
    assert abs(axis[0]) < 0.1, f"X 축이 아닌데 X 성분이 크다: {axis}"
    assert abs(axis[1]) > 0.9, f"Y 축이어야 한다: {axis}"


def test_axis_sign_differs_between_arms():
    """★ 두 팔의 축 부호가 반대다. 숫자를 박으면 한쪽이 반대로 돈다."""
    signs = {}
    for path in CONFIGS:
        if not path.exists():
            pytest.skip(f"{path.name} 없음")
        cfg = ArmConfig.load(path)
        axis, _, _ = sim.measure_wrist_axis(cfg=cfg, ik=sim.ArmIK(cfg))
        signs[path.stem] = float(np.sign(axis[1]))
    assert len(set(signs.values())) == 2, \
        f"부호가 같다 — 이 테스트의 전제가 바뀌었으면 주석도 고칠 것: {signs}"


def test_roll_tracks_one_to_one_over_full_range(arm):
    """요청한 롤만큼 끝관절이 정확히 돌아야 한다. 위치는 안 흔들려야 한다."""
    cfg, ik = arm
    axis, lo, hi = sim.measure_wrist_axis(ik, cfg)
    ik.set_q(cfg.home)
    home_ee = ik.fk()
    home_last = float(cfg.home[-1])

    for deg in (-178, -120, -90, -45, 45, 90, 120, 178):
        want = float(np.clip(np.radians(deg), lo, hi))
        target = home_ee.copy()
        target[:3, :3] = home_ee[:3, :3] @ rotvec_to_rotation(axis * want)
        ik.set_q(cfg.home)
        for _ in range(300):
            q, _ = ik.solve(target)
        ee = ik.fk()

        Rerr = ee[:3, :3].T @ target[:3, :3]
        ori_err = np.degrees(np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1)))
        pos_err = np.linalg.norm(ee[:3, 3] - target[:3, 3]) * 1000
        rel = float(q[-1]) - home_last

        assert ori_err < 0.5, f"{deg}° 요청에서 자세오차 {ori_err:.2f}°"
        assert pos_err < 0.5, f"{deg}° 요청에서 위치가 {pos_err:.2f}mm 흔들렸다"
        assert abs(rel - want) < np.radians(0.5), \
            f"{deg}° 요청인데 끝관절은 {np.degrees(rel):.1f}° 돌았다"


def test_roll_range_has_no_dead_zone(arm):
    """롤 범위는 홈 자세 기준 상대값이어야 한다.

    대칭(±min)으로 잡으면 끝관절이 홈에서 이미 돌아가 있는 만큼 한쪽 끝이
    도달 불가가 되어 데드존이 생긴다.
    """
    cfg, ik = arm
    _, lo, hi = sim.measure_wrist_axis(ik, cfg)
    home_last = float(cfg.home[-1])
    assert np.isclose(lo, cfg.lower[-1] - home_last, atol=1e-9)
    assert np.isclose(hi, cfg.upper[-1] - home_last, atol=1e-9)
    assert hi > lo


def test_no_roll_when_last_joint_does_not_rotate_ee(arm):
    """끝관절이 EE 자세를 안 바꾸는 팔에서는 롤을 아예 끈다 (0 벡터)."""
    cfg, ik = arm

    class Flat:                     # EE 가 절대 안 도는 가짜 IK
        def set_q(self, q): pass
        def fk(self): return np.eye(4)

    axis, lo, hi = sim.measure_wrist_axis(Flat(), cfg)
    assert np.allclose(axis, 0.0) and lo == 0.0 and hi == 0.0
