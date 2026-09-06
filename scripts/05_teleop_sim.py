#!/usr/bin/env python
"""5단계: Quest 2 → IK → 3D 시각화 실시간 텔레옵 (하드웨어 없음).

Quest 오른손 컨트롤러로 가상의 RPO 팔을 실제로 조종해 본다. 로봇을 붙이기 전에
좌표계 방향, 스케일, 클러치 감각을 눈으로 확인하는 단계다.

    Quest 2 (WebXR 90Hz)
      → QuestXRSource            web/index.html 이 보내는 원본 입력
      → ClutchState              grip 클러치 + RUB→FLU 변환 + 스케일 0.76
      → 목표 EE pose             시작 자세 기준 상대 변위/회전
      → placo IK (iters=5)       0.03 ms
      → Meshcat 3D 뷰            브라우저에서 팔이 따라 움직임

조작 (오른손 컨트롤러):
    Grip(중지) 꾹    팔 추종 활성화. 떼면 그 자리에 정지, 다시 잡으면 이어서 조작
    Trigger(검지)    손 쥐는 정도 0~1 (AmazingHand grasp, A가 ON일 때만)
    A                AmazingHand grasp on/off 토글 (Quest 오른쪽 홈 버튼). 끄면 마지막 쥐기 유지
    B                소프트웨어 정지 래치(토크를 유지한 채 HOLD)
    썸스틱 X         손목 롤(elbow_yaw) 오프셋

사용법:
    .venv/bin/python scripts/05_teleop_sim.py
    .venv/bin/python scripts/05_teleop_sim.py --scale 0.6 --rate 30
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import placo
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpo_teleop.arm_config import ArmConfig  # noqa: E402
from rpo_teleop.certs import get_local_ip, list_local_ips  # noqa: E402
from rpo_teleop.grasp_enable import gated_grasp, toggle_grasp_enable  # noqa: E402
from rpo_teleop.hand_model import HandModel, grasp_to_servo  # noqa: E402
from rpo_teleop.jetson_link import (  # noqa: E402
    STATE_TRIP, JetsonBackend, check_robot_match, home_reached)
from rpo_teleop import profiles  # noqa: E402
from rpo_teleop.arm_visual import ArmVisual  # noqa: E402
from rpo_teleop.transforms import ClutchState, rotvec_to_rotation  # noqa: E402
from rpo_teleop.xr_server import build_joint_state  # noqa: E402
from rpo_teleop.xr_source import QuestXRSource  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "superarm_j1_j2_jetson.json"


def latch_software_stop(
    latched: bool, pressed: bool, was_pressed: bool
) -> tuple[bool, bool, bool]:
    """B 상승 엣지에서 소프트웨어 HOLD를 한 번만 래치한다.

    Returns: ``(latched, was_pressed, newly_latched)``. 한 번 걸리면 해제
    입력은 없고 프로세스를 재시작해야 한다.
    """
    newly_latched = bool(pressed and not was_pressed and not latched)
    return bool(latched or newly_latched), bool(pressed), newly_latched


def hold_current_position(motors, dof: int) -> np.ndarray:
    """현재 관절각을 전송하고 HOLD 모드로 토크를 유지한다.

    B 소프트웨어 정지에서는 ``disable()``을 절대 호출하지 않는다.
    소자는 CAN 피드백 유실 같은 실제 고장 경로에만 남겨 둔다.
    """
    actual = np.asarray(motors.read_positions(), dtype=float)
    motors.hold()
    motors.write_positions(actual, dq=np.zeros(dof), engaged=False)
    return actual


class ArmIK:
    """placo 기반 IK — 위치는 hard, 자세는 soft.

    5-DOF 팔로는 6-DOF 목표를 정확히 맞출 수 없다. 자유도 예산이 1 모자란다.
    두 목표를 같은 가중치 QP(add_frame_task)에 넣으면 둘 다 어중간해진다 (실측):

        pos_w=1     ori_w=1  → 위치 62.65 mm, 자세  0.7°
        pos_w=100   ori_w=1  → 위치 13.72 mm, 자세 15.3°
        pos_w=10000 ori_w=1  → 위치  0.17 mm, 자세 19.3°   ← 채택

    위치를 hard constraint 로 걸면 0.00 mm 가 나오지만, 도달 불가능한 목표가 들어오는
    순간 QP 가 불능이 되어 예외로 떨어진다. 워크스페이스는 관절 한계 때문에 완전한
    구가 아니라 부분 껍질이라, 반경 클램프만으로는 도달 가능성을 보장할 수 없다(실측).
    가중치 10000:1 의 soft 는 정확도가 사실상 같으면서(0.17 mm) 도달 불가 목표에서도
    예외 없이 "갈 수 있는 가장 가까운 자세"로 수렴한다. 텔레옵에는 이쪽이 안전하다.

    자세 오차 ~19° 는 버그가 아니라 5-DOF 팔로 6-DOF 목표를 쫓을 때의 물리적 한계다.
    """

    def __init__(self, cfg: ArmConfig, pos_weight: float = 10000.0,
                 ori_weight: float = 1.0, iters: int = 5):
        self.cfg = cfg
        self.robot = placo.RobotWrapper(cfg.urdf_path)
        self.names = list(self.robot.joint_names())
        self.frame = cfg.ee_frame
        self.iters = iters

        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        # 속도 제한이 없으면 관절 한계가 정확히 ±180° 인 팔에서 솔버가 +179°↔-179° 를
        # 자유롭게 오간다. 기구학적으로는 같은 자세라 QP 는 만족하지만, 실물 모터에는
        # 한 프레임에 360° 회전하라는 명령이 나간다. 실측: 제한 없으면 한 스텝 최대
        # 변화 360.0°, 추종 p95 311 mm → 제한 켜면 2.5°, p95 44.8 mm.
        #
        # ★ dt 를 iters 로 나눈다. placo 는 solve() 한 번을 한 스텝으로 보고
        #   dt × 속도한계 만큼만 움직이도록 제한한다. 한 프레임에 iters 번 돌면
        #   프레임당 실제 이동량이 정확히 iters 배가 된다 (실측: iters=5 에서
        #   허용 7.20°/프레임 대비 36.00°, 정확히 5.0배). 나눠주면 7.20° 로 맞는다.
        #   안 나누면 시뮬과 기록 데이터에 실물이 못 내는 속도가 섞인다.
        self.solver.dt = cfg.control_dt / max(iters, 1)
        self.solver.enable_velocity_limits(True)
        self.pos_task = self.solver.add_position_task(cfg.ee_frame, np.zeros(3))
        self.pos_task.configure("ee_pos", "soft", pos_weight)
        self.ori_task = self.solver.add_orientation_task(cfg.ee_frame, np.eye(3))
        self.ori_task.configure("ee_ori", "soft", ori_weight)

        self.set_q(cfg.home)

    def set_q(self, q: np.ndarray) -> None:
        for name, val in zip(self.names, q, strict=True):
            self.robot.set_joint(name, float(val))
        self.robot.update_kinematics()

    @property
    def q(self) -> np.ndarray:
        return np.array([self.robot.get_joint(n) for n in self.names])

    def fk(self, frame: str | None = None) -> np.ndarray:
        return self.robot.get_T_world_frame(frame or self.frame)

    def clamp_to_workspace(self, pos: np.ndarray) -> tuple[np.ndarray, bool]:
        """어깨 기준 리치 안으로 목표를 끌어당긴다.

        soft task 라 클램프가 없어도 예외는 안 나지만, 클램프해 두면 도달 한계에서의
        거동이 예측 가능해지고(어깨-목표 직선상으로 뻗음) 누적 변위가 무한정 커지는
        것도 막는다. LeRobot 의 EEBoundsAndSafety 가 하는 일과 같다.
        """
        return self.cfg.clamp_position(pos)

    def solve(self, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """IK 를 푼다. Returns (q, clamped)."""
        pos, clamped = self.clamp_to_workspace(target[:3, 3])
        self.pos_task.target_world = pos
        self.ori_task.R_world_frame = target[:3, :3]
        q_backup = self.q
        try:
            for _ in range(self.iters):
                self.solver.solve(True)
                self.robot.update_kinematics()
        except Exception:
            # soft task 에서는 사실상 오지 않는 경로. 만약을 대비해 직전 자세 유지
            # (로봇이 튀는 것보다 멈추는 쪽이 안전하다)
            self.set_q(q_backup)
            return q_backup, True
        return self.q, clamped


def measure_wrist_axis(ik: "ArmIK", cfg: ArmConfig) -> tuple[np.ndarray, float, float]:
    """끝관절을 조금 돌려 **EE 로컬 회전축**을 실측한다.

    ★ 축을 코드에 박으면 안 된다. 팔마다 다르고, 심지어 **부호가 반대**다.

        config/arm.json       joint5  EE로컬 축 = [0, -1, 0]
        config/arm_temp.json  joint3  EE로컬 축 = [0, +1, 0]   ← 부호 반대

      예전 구현은 X축 [w, 0, 0] 을 썼는데 실제 축은 둘 다 Y 였다. 그래서 팔이
      **낼 수 없는 회전을 요청**하고 있었다. 자세는 soft task(가중치 1 : 위치
      10000)라 예외 없이 "갈 수 있는 가장 가까운 자세"로 수렴하는데, 결과가
      엉망이었다 (실측):

          전체 팔  : 자세오차 최대 117.7°, 음의 방향이 사실상 죽음
          임시 팔  : 끝관절이 **아예 안 움직임**, 오차 177.7°

      URDF 가 진실이므로 기동할 때 한 번 재서 쓴다.

    Returns:
        (축 단위벡터, 롤 하한 rad, 롤 상한 rad)
        하한/상한은 **홈 자세 기준 상대값**이다. 끝관절이 홈에서 이미 얼마쯤
        돌아가 있으면 양쪽으로 갈 수 있는 양이 다르다. 대칭으로 잡으면 한쪽에
        데드존이 생긴다.
    """
    q0 = np.asarray(cfg.home, dtype=float)
    ik.set_q(q0)
    T0 = ik.fk()
    q1 = q0.copy()
    q1[-1] += np.radians(5.0)
    ik.set_q(q1)
    T1 = ik.fk()
    ik.set_q(q0)

    R = T0[:3, :3].T @ T1[:3, :3]
    ang = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    if ang < np.radians(0.5):
        # 끝관절이 EE 자세를 거의 안 바꾸는 팔. 손목 롤을 줄 수 없다.
        return np.zeros(3), 0.0, 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    axis = axis / np.linalg.norm(axis)

    lo = float(cfg.lower[-1] - q0[-1])
    hi = float(cfg.upper[-1] - q0[-1])
    return axis, lo, hi


def view_matrix4(view3: np.ndarray) -> list[float]:
    """3x3 뷰 변환 → 4x4 행렬을 row-major 16개로. three.js Matrix4.set() 순서."""
    m = np.eye(4)
    m[:3, :3] = view3
    return [round(float(v), 6) for v in m.reshape(-1)]


def idle_state(ik: "ArmIK", cfg: ArmConfig, scale: float,
               visual: ArmVisual | None = None, view: np.ndarray | None = None) -> dict:
    """Quest 미연결 시 대시보드/VR 에 보낼 최소 상태."""
    q = ik.q
    ee = ik.fk()
    state = build_joint_state(cfg, q, clamped=False)
    if view is not None:
        state["view_matrix"] = view_matrix4(view)
    if visual is not None:
        state["geometry"] = visual.geometry
        state["links"] = visual.poses()
        state["ee_point"] = visual.point(ee[:3, 3])
    state.update({
        "robot": {"name": Path(cfg.urdf_path).name, "dof": cfg.dof,
                  "ee_frame": cfg.ee_frame, "scale": scale,
                  "position_rank": cfg.position_rank},
        "engaged": False,
        "ee": {"x": float(ee[0, 3]), "y": float(ee[1, 3]), "z": float(ee[2, 3])},
        "reach": {"current": float(np.linalg.norm(ee[:3, 3] - cfg.shoulder)),
                  "min": cfg.min_reach, "max": cfg.max_reach,
                  "span_max": cfg.max_reach * 1.15},
        "waiting": True,
    })
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description="Quest 2 → IK → Meshcat 실시간 텔레옵")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="scripts/06_setup_urdf.py 가 만든 설정 파일")
    ap.add_argument("--urdf", type=Path, default=None,
                    help="설정 파일 대신 URDF 를 직접 지정 (즉석 측정)")
    ap.add_argument("--profile", type=str, default="jetson",
                    choices=sorted(profiles.PROFILE_SLOTS),
                    help="포트 블록. 실물=jetson, Isaac Sim=isaac. "
                         "두 갈래를 동시에 띄우려면 반드시 달라야 한다")
    ap.add_argument("--port", type=int, default=None,
                    help="웹 포트 직접 지정 (기본: 프로파일에서 결정)")
    ap.add_argument("--ip", type=str, default=None)
    ap.add_argument("--rate", type=float, default=30.0, help="제어 루프 Hz")
    ap.add_argument("--scale", type=float, default=None,
                    help="사람 손 변위 → 로봇 EE 변위 배율 (기본: 설정에서 자동 산출)")
    ap.add_argument("--ori-weight", type=float, default=1.0,
                    help="자세 soft task 가중치 (위치는 hard 라 영향 없음)")
    ap.add_argument("--pos-weight", type=float, default=10000.0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--base-yaw", type=float, default=0.0,
                    help="WebXR→로봇 매핑의 수직축 보정각(도). 팔을 놓은 방향에 맞춘다")
    ap.add_argument("--mirror", action="store_true",
                    help="좌우 반사. 앞뒤는 맞는데 좌우만 뒤집혀 보일 때")
    ap.add_argument("--meshcat", action="store_true",
                    help="Meshcat 3D 뷰도 띄운다. 대시보드(/dashboard)에 이미 3D 뷰가 있어\n"
                         "보통은 불필요하고, 포트 7000 이 macOS AirPlay 와 충돌해 번호가 계속 바뀐다")
    ap.add_argument("--no-hand", action="store_true", help="손을 표시/제어하지 않음")
    ap.add_argument("--motors", type=str, default="none",
                    choices=("none", "jetson"),
                    help="모터 출력. none=시뮬만, jetson=UDP 로 젯슨(또는 가짜 젯슨)에 지령")
    ap.add_argument("--jetson-host", type=str, default=None,
                    help="젯슨 주소. 생략하면 비컨(UDP 5007)으로 자동 탐색")
    ap.add_argument("--no-home-on-xr-start", action="store_true",
                    help="Start XR 직후 실물/시뮬 HOME 정렬을 생략한다")
    ap.add_argument("--disable-home-button", action="store_true",
                    help="오른손 A의 HOME 이동을 비활성화한다. A는 AmazingHand grasp on/off 토글로 남는다")
    ap.add_argument("--no-viz", action="store_true", help=argparse.SUPPRESS)  # 하위호환
    args = ap.parse_args()

    if args.urdf is not None:
        print(f"  URDF 즉석 측정 중… ({args.urdf.name})", flush=True)
        cfg = ArmConfig.from_urdf(args.urdf)
    elif args.config.exists():
        cfg = ArmConfig.load(args.config)
    else:
        print(f"❌ 설정 파일이 없습니다: {args.config}\n"
              f"   먼저 실행하세요: .venv/bin/python scripts/06_setup_urdf.py --urdf <파일>",
              file=sys.stderr)
        return 1

    # ── 포트 블록 ────────────────────────────────────────────────────
    #   같은 저장소에서 실물(jetson)과 Isaac Sim(isaac)이 동시에 돈다.
    #   포트가 겹치면 늦게 뜬 쪽이 "Address already in use" 로 죽거나,
    #   UDP 는 조용히 패킷을 나눠 먹어서 원인 찾기가 지독해진다.
    pf = profiles.ports(args.profile)
    web_port = args.port if args.port is not None else pf.web
    problems = profiles.check_free(pf, need_web=(args.port is None),
                                   need_udp=(args.motors == "jetson"))
    if problems:
        print(profiles.explain_conflict(pf, problems), file=sys.stderr)
        return 1

    # 화면에 찍을 "지금 무엇을 움직이나". 포트가 아니라 이걸 알고 싶어한다.
    # ★ 프로파일마다 받는 쪽이 다르다. jetson 만 실물이다.
    #   test 프로파일에 "Isaac Sim" 이라고 찍히면 사람이 헷갈린다.
    RECEIVER_NAMES = {"jetson": "실물 젯슨", "isaac": "Isaac Sim", "test": "테스트 대상"}
    if args.motors == "none":
        receiver = {"profile": args.profile, "kind": "sim", "label": "시뮬만 (모터 없음)"}
    else:
        where = args.jetson_host or "탐색 중"
        kind = "real" if args.profile == "jetson" else "sim"
        name = RECEIVER_NAMES.get(args.profile, args.profile)
        receiver = {"profile": args.profile, "kind": kind,
                    "label": f"{name} {where}"}

    scale = args.scale if args.scale is not None else cfg.position_scale
    ik = ArmIK(cfg, pos_weight=args.pos_weight, ori_weight=args.ori_weight, iters=args.iters)
    home_ee = ik.fk()
    # ★ 손목 롤 축은 배너에서도 쓰므로 여기서 잰다 (ik 가 만들어진 직후).
    #   팔마다 다르고 부호도 반대라 숫자를 박을 수 없다.
    roll_axis, roll_lo, roll_hi = measure_wrist_axis(ik, cfg)
    hand = None
    if not args.no_hand:
        try:
            hand = HandModel()
            hand.set_grasp(0.0)
        except Exception as exc:
            print(f"  ⚠️  손 모델 로드 실패 ({exc}) — 팔만 표시합니다", flush=True)
    visual = ArmVisual(ik.robot, cfg.urdf_path, hand=hand)

    viz = None
    viz_url = None
    if args.meshcat and not args.no_viz:
        viz = robot_viz(ik.robot)
        viz.display(ik.robot.state.q)
        # macOS 는 AirPlay 수신기가 7000 을 쓰고 있어 meshcat 이 7001 등으로 폴백한다.
        # 고정 포트를 안내하면 엉뚱한 주소를 보게 되므로 실제 URL 을 읽어서 출력한다.
        try:
            viz_url = viz.viewer.url()
        except Exception:
            viz_url = "(meshcat URL 확인 실패 — 터미널 상단 로그를 보세요)"

    # 모터 백엔드. --motors none 이면 붙이지 않는다 (지금까지의 시뮬 동작 그대로).
    #
    # ★ 안전 제한(관절한계·속도·소프트스타트·이격·워치독)은 여기서 걸지 않는다.
    #   전부 젯슨이 건다. 상태를 갖는 적분기가 양쪽에 있으면 두 지령이 서서히
    #   갈라지고 어느 쪽이 진짜인지 알 수 없게 된다 (handoff/13_JETSON_REPLY2.txt §7).
    motors = None
    if args.motors == "jetson":
        motors = JetsonBackend(n_joints=cfg.dof, host=args.jetson_host,
                               cmd_port=pf.cmd, state_port=pf.state,
                               beacon_port=pf.beacon,
                               discover=args.jetson_host is None)
        motors.connect()
        motors.hold()          # 연결 절차 2) — 먼저 HOLD 로 링크만 확인한다

    # 메시는 지금 쓰는 URDF 옆 meshes/ 에서 서빙한다 (임시 팔은 폴더가 다르다)
    source = QuestXRSource(port=web_port, ip=args.ip, use_grip_space=True,
                           arm_mesh_dir=Path(cfg.urdf_path).parent / "meshes")
    if motors is not None:
        def on_dashboard_cmd(data: dict) -> None:
            """대시보드/HUD 버튼.

            ★ 트립 해제는 사람이 직접 눌러야만 일어난다. 자동 재시도 루프에
              넣으면 "자동 복구 금지" 원칙이 무너진다 (13_JETSON_REPLY2.txt §1-D).
            """
            if data.get("action") == "clear_trip":
                motors.clear_trip()
                print("  ⏻ 트립 해제 요청 (대시보드)", flush=True)

        source.subscribe_command(on_dashboard_cmd)

    source.start()
    clutch = ClutchState(position_scale=scale, base_yaw_deg=args.base_yaw,
                         mirror=args.mirror)

    def clamp_delta(delta: np.ndarray) -> np.ndarray:
        """누적 변위를 워크스페이스 안으로. 클러치가 재앵커까지 처리한다."""
        return ik.clamp_to_workspace(home_ee[:3, 3] + delta)[0] - home_ee[:3, 3]

    print("=" * 74)
    print("  Quest 2 → IK → 3D 시각화 실시간 텔레옵")
    print("=" * 74)
    all_ips = list_local_ips()
    print(f"  Quest 접속  : {source.url}")
    for extra in all_ips[1:]:
        print(f"                https://{extra}:{args.port}   (다른 인터페이스)")
    if viz is not None:
        print(f"  3D 뷰       : {viz_url}")
    print(f"  대시보드    : {source.url}/dashboard   (관절 한계 실시간 표시)")
    print(f"  프로파일    : {args.profile}   웹 {web_port}  UDP 지령/상태/비컨 "
          f"{pf.cmd}/{pf.state}/{pf.beacon}")
    print(f"  로봇        : {Path(cfg.urdf_path).name}  {cfg.dof}-DOF  EE={cfg.ee_frame}")
    print(f"  제어 주기   : {args.rate:.0f} Hz   스케일 {scale:.3f}   "
          f"리치 {cfg.min_reach:.2f}~{cfg.max_reach:.2f} m   IK iters {args.iters}")
    if motors is not None:
        where = args.jetson_host or f"비컨으로 자동 탐색 중 (UDP {pf.beacon})"
        print(f"  모터        : 젯슨 {where}")
        print("                안전 제한(속도·한계·워치독)은 전부 젯슨이 겁니다.")
        print("                클러치를 처음 잡아야 RUN 이 나갑니다.")
    else:
        print("  모터        : 없음 (시뮬만). 붙이려면 --motors jetson")
    if cfg.position_rank < 3:
        print("-" * 74)
        print(f"  ⚠️  이 팔은 위치 자유도가 {cfg.position_rank}/3 입니다 (자코비안 랭크).")
        print("      손끝이 곡면 위에서만 움직입니다. 조종할 때:")
        print("        · 가리키는 방향 바꾸기  → 정상 동작 (전달률 95%)")
        print("        · 안팎으로 밀고 당기기  → 반응 없음 (전달률 1%)  ← 고장 아님")
        print("      관절 수가 아니라 축 배치의 문제라 설정으로는 못 고칩니다.")
    print("-" * 74)
    if roll_hi > roll_lo:
        print(f"  손목 롤      : 축 {np.round(roll_axis, 3)} (EE 로컬, URDF 실측)  "
              f"범위 {np.degrees(roll_lo):+.0f}° ~ {np.degrees(roll_hi):+.0f}°")
    else:
        print("  손목 롤      : 없음 (끝관절이 EE 자세를 안 바꿉니다)")
    print("  오른손 Grip 을 꾹 누른 채 팔을 움직이세요. 떼면 그 자리에 멈춥니다.")
    home_label = "A=비활성(홈)" if args.disable_home_button else "A=홈 복귀"
    print(f"  {home_label}   B=소프트웨어 정지(HOLD)   썸스틱X=손목 롤")
    print("  B를 누르면 토크를 유지한 채 정지하며, 재가동은 프로그램 재시작으로만 가능합니다.")
    if hand is not None:
        print(
            "  손: 트리거=쥐는 정도(0~1)  A(홈 버튼)=grasp on/off  "
            f"서보 {hand.n_servos}개 / 관절 {hand.n_joints}개"
        )
    print("  [좌우가 반대로 느껴지면] 왼손 X=미러 토글, 왼손 Y=yaw +90°")
    print(f"  현재 매핑: yaw {args.base_yaw:.0f}°  mirror {'ON' if args.mirror else 'off'}")
    if args.mirror:
        print()
        print("  ⚠️  미러가 켜져 있습니다 — 화면이 실물의 거울상입니다.")
        print("      이 손은 오른손(amazing_hand_right)인데 화면에는 왼손으로 보입니다.")
        print("      반사(det=-1)는 회전이 아니라서 좌우 손이 구분되는 로봇에는 맞지 않습니다.")
        print("      엄지 위치·접근 방향이 실제와 반대로 보이므로 데이터 수집 전에")
        print("      yaw 보정(--base-yaw 0/90/180/270)으로 대체하시기를 권합니다.")
        print("      VR 안에서는 왼손 Y 로 yaw 를, 왼손 X 로 미러를 끌 수 있습니다.")
    print("  Ctrl+C 로 종료")
    print("=" * 74, flush=True)

    source.publish_state(idle_state(ik, cfg, scale, visual, np.linalg.inv(clutch.mapping)))

    running = True

    def on_sigint(_s, _f):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    wrist_roll = 0.0
    prev_left_x = prev_left_y = False
    prev_right_a = False
    prev_right_a_raw = False
    prev_right_b = False
    software_stopped = False
    grasp_enabled = True
    held_grasp = 0.0
    last_print = 0.0
    # 모터: 실물 관절각으로 IK 시작점을 맞췄는가 / RUN 을 내보낼 자격이 있는가
    motor_synced = False
    motor_armed = False
    dof_warned = False
    q_prev_cmd = None
    # 핫스팟/DHCP 환경에서는 IP 가 실행 중에도 바뀐다. 그러면 헤드셋에서 열어둔
    # 주소가 죽는데, 화면을 보고 있지 않으면 "왜 안 되지"로만 남는다. 주기적으로 확인한다.
    startup_ip = source.ip
    last_ip_check = time.time()
    loop_times: list[float] = []
    ik_times: list[float] = []
    connected_once = False
    # 실물은 프로세스 실행만으로 움직이지 않는다. 조종자가 Quest에서 Start XR을
    # 눌러 세션을 연 시점을 시작 승인으로 보고, 그때 한 번 HOME으로 정렬한다.
    # 시뮬 전용 실행은 기존처럼 cfg.home에서 이미 시작하므로 별도 절차가 없다.
    home_on_xr_start = motors is not None and not args.no_home_on_xr_start
    homing = False
    period = 1.0 / args.rate

    while running:
        t_loop = time.perf_counter()
        frame = source.latest

        if frame is None or frame.right is None:
            # Quest 가 붙기 전에도 대시보드에서 현재 자세와 관절 한계를 볼 수 있어야
            # 하므로 최소 상태는 계속 내보낸다.
            st = idle_state(ik, cfg, scale, visual, np.linalg.inv(clutch.mapping))
            # 연결 절차 2) — 헤드셋을 쓰기 전에 HOLD 로 링크부터 확인한다.
            #   여기서 안 보내면 젯슨은 Mac 이 있는지조차 모르고, 사람이 헤드셋을
            #   쓴 뒤에야 링크 문제를 발견하게 된다.
            if motors is not None:
                motor_armed = False
                motors.hold()
                motors.write_positions(motors.read_positions(),
                                       dq=np.zeros(cfg.dof), engaged=False)
                q_prev_cmd = None      # 무장하면 여기서부터 다시 센다
                jst = motors.state
                # ★ 반드시 쓰기 전에 검사한다. 관절 수가 다르면 IK 를 맞추려다
                #   zip(strict=True) 가 터져 서버가 반쯤 뜬 채 죽는다.
                mismatch = check_robot_match(jst, cfg.urdf_path, cfg.dof)
                if mismatch:
                    st["dof_mismatch"] = mismatch
                    if not dof_warned:
                        dof_warned = True
                        print("\n" + "=" * 74, flush=True)
                        print("  ❌ 로봇이 안 맞습니다 — 모터를 움직이지 않습니다",
                              flush=True)
                        print(f"     {mismatch}", flush=True)
                        print("     임시 팔(3-DOF)이면  ./run.sh --temp --motors jetson",
                              flush=True)
                        print("     전체 팔(5-DOF)이면  ./run.sh --motors jetson",
                              flush=True)
                        print("=" * 74 + "\n", flush=True)
                elif jst is not None and not motor_synced:
                    ik.set_q(np.asarray(jst.q, dtype=float)[: cfg.dof])
                    home_ee = ik.fk()
                    clutch.reset()
                    motor_synced = True
                    print(f"  ⟳ 실물 관절각으로 동기: {np.round(np.degrees(jst.q), 1)}°",
                          flush=True)
                st["motors"] = motors.summary()
                # Quest 가 붙기 전에도 실물 관절각을 봐야 한다. 사전 점검에서
                # "화면의 팔이 실물과 같은 자세인가" 를 여기서 확인한다.
                st["q_act"] = [round(float(v), 6) for v in motors.read_positions()]
                st["q_cmd"] = [round(float(v), 6) for v in ik.q]
                if motors.address is not None:
                    receiver["label"] = (f"{RECEIVER_NAMES.get(args.profile, args.profile)} "
                                         f"{motors.address[0]}")
            st["receiver"] = receiver
            # ★ "한 번도 안 붙음" 과 "붙었다가 끊김" 은 완전히 다른 상황이다.
            #   헤드셋을 벗거나 슬립에 들면 WebXR 세션이 끊기는데, 예전에는
            #   connected_once 가 True 라 아무 말도 안 하고 조용히 멈췄다.
            #   조종자에겐 "멀쩡해 보이는데 안 움직인다" 로만 보인다
            #   (시뮬 담당 실측: 37분 무음). 사람이 seq 증가율로 진단하게 두면 안 된다.
            st["quest"] = "lost" if connected_once else "waiting"
            source.publish_state(st)
            if time.time() - last_print > 2.0:
                if connected_once:
                    print("  ⚠️  Quest 연결 끊김 — 헤드셋에서 [Start XR] 을 다시 누르세요",
                          flush=True)
                else:
                    print("  … Quest 접속 대기 중 "
                          f"({source.url} → [Start XR])", flush=True)
                last_print = time.time()
            time.sleep(0.1)
            continue

        if not connected_once:
            connected_once = True
            homing = home_on_xr_start
            if homing:
                print("  ✅ Quest 연결됨 — J1/J2와 시뮬을 HOME으로 정렬합니다", flush=True)
            else:
                print("  ✅ Quest 연결됨 — Grip 을 눌러 조작을 시작하세요", flush=True)

        right = frame.right

        # 왼손 버튼으로 좌표 매핑을 실시간 조절.
        # 헤드셋을 쓴 채 재시작 없이 맞춰볼 수 있어야 한다.
        left = frame.left
        if left is not None:
            if left.primary and not prev_left_x:      # X — 미러 토글
                clutch.mirror = not clutch.mirror
                clutch.reset()
                print(f"  ↔ 미러 {'ON' if clutch.mirror else 'off'} "
                      f"(yaw {clutch.base_yaw_deg:.0f}°)", flush=True)
            if left.secondary and not prev_left_y:    # Y — yaw 90° 회전
                clutch.base_yaw_deg = (clutch.base_yaw_deg + 90.0) % 360.0
                clutch.reset()
                print(f"  ↻ yaw {clutch.base_yaw_deg:.0f}° "
                      f"(미러 {'ON' if clutch.mirror else 'off'})", flush=True)
            prev_left_x, prev_left_y = left.primary, left.secondary

        # 버튼 처리. B 는 다른 입력보다 먼저 처리하고, 한 번 걸리면
        # 재시작 전까지 절대 재무장하지 않는다.
        software_stopped, prev_right_b, newly_stopped = latch_software_stop(
            software_stopped, bool(right.secondary), prev_right_b
        )
        if newly_stopped:
            motor_armed = False
            q_prev_cmd = None
            clutch.reset()
            if motors is not None:
                hold_current_position(motors, cfg.dof)
            print("\n  ⏸ B 소프트웨어 정지: 토크 유지 HOLD, 재시작 전까지 재무장 금지\n",
                  flush=True)

        if software_stopped:
            st = idle_state(ik, cfg, scale, visual, np.linalg.inv(clutch.mapping))
            st.update({
                "quest": "ok",
                "software_stop": True,
                "receiver": receiver,
                "motors": None if motors is None else motors.summary(),
            })
            if motors is not None:
                hold_current_position(motors, cfg.dof)
            source.publish_state(st)
            time.sleep(period)
            continue

        a_raw = bool(right.primary)
        a_pressed = a_raw and not args.disable_home_button
        if a_pressed and not prev_right_a:  # A — 실물/시뮬 동기식 홈 복귀
            ik.set_q(cfg.home)
            clutch.reset()
            wrist_roll = 0.0
            homing = motors is not None
            motor_armed = False
            q_prev_cmd = None
            if homing:
                print("  ⌂ A HOME 요청 — J1/J2를 제한 속도로 이동합니다", flush=True)
        prev_right_a = a_pressed

        # 썸스틱 X → 손목 롤 오프셋. EE 자세에 더해 IK 가 풀도록 한다.
        # (elbow_yaw 를 직접 지령하지 않고 EE pose 추상화 안에 유지)
        if abs(right.stick[0]) > 0.15 and roll_hi > roll_lo:
            # ★ ±90° 를 박아두면 사용자가 요청한 ±178° 가 안 나온다. 그리고
            #   대칭으로 잡으면 홈에서 이미 돌아가 있는 만큼 한쪽에 데드존이 생긴다.
            wrist_roll = float(np.clip(wrist_roll + right.stick[0] * 0.03,
                                       roll_lo, roll_hi))

        # 트리거(0~1) → 손 쥐는 정도.
        # ★ 반드시 서보 공간에서 보간한다. 관절 공간 보간은 중간에서 최대 11.44°
        #   어긋난다 (직접 검증). grasp_to_servo() 가 그 처리를 한다.
        # B 는 팔 HOLD 에 쓰이므로 손 프리셋에 겹쳐 쓰지 않는다.
        # 오른손 A(홈 버튼)는 AmazingHand grasp 제어 on/off 토글이다. 꺼져 있으면
        # 트리거를 무시하고 마지막 쥐기 값을 유지한다. 홈 이동은
        # --disable-home-button 으로 따로 막는다.
        if hand is not None:
            prev_grasp_enabled = grasp_enabled
            grasp_enabled, prev_right_a_raw = toggle_grasp_enable(
                grasp_enabled, a_raw, prev_right_a_raw
            )
            if grasp_enabled != prev_grasp_enabled:
                print(
                    f"  AmazingHand grasp {'ON' if grasp_enabled else 'OFF'}",
                    flush=True,
                )
        else:
            prev_right_a_raw = a_raw
        grasp = gated_grasp(float(right.trigger), grasp_enabled, held_grasp)
        held_grasp = grasp
        if hand is not None:
            hand.set_grasp(grasp)

        engaged, delta_pos, rel_rotvec = clutch.update(
            right.position, right.rotation_matrix, right.grip, clamp_fn=clamp_delta
        )

        # 목표 EE pose = 홈 자세 기준 + 클러치 변위/회전 + 손목 롤
        target = np.eye(4)
        target[:3, 3] = home_ee[:3, 3] + delta_pos
        # ★ 회전 적용 순서 주의
        #   클러치가 주는 rel_rotvec 은 **월드 기준** 상대 회전이다 (R_now · R_ref⁻¹).
        #   따라서 왼쪽에서 곱해야 한다. 오른쪽에 곱하면 바디 기준으로 해석되어
        #   회전축이 home 자세만큼 틀어진다 (실측: 축이 170.4° 어긋나 손목을 돌리면
        #   로봇 손이 거의 반대로 돈다).
        #   반면 손목 롤(썸스틱)은 EE 자기 축 기준이라 오른쪽에 곱하는 게 맞다.
        # ★ 축은 measure_wrist_axis() 가 URDF 에서 실측한 것이다. 박으면 안 된다
        #   — 팔마다 다르고 부호까지 반대다 (그 함수 주석 참고).
        roll = rotvec_to_rotation(roll_axis * wrist_roll)
        target[:3, :3] = rotvec_to_rotation(rel_rotvec) @ home_ee[:3, :3] @ roll

        t_ik = time.perf_counter()
        q, clamped = ik.solve(target)
        ik_times.append(time.perf_counter() - t_ik)

        # HOME 중에는 Quest 변위보다 cfg.home 관절 목표가 우선이다. 실제 속도와
        # 관절 한계는 젯슨 안전계층이 적용하며, 완료 전에는 일반 Grip 무장을 막는다.
        if homing:
            q = np.asarray(cfg.home, dtype=float).copy()
            ik.set_q(q)

        # ── 모터 출력 ────────────────────────────────────────────────
        if motors is not None:
            jst = motors.state
            # ★ 관절 수가 다르면 아무것도 하지 않는다. 지령을 보내면 엉뚱한
            #   관절이 움직이고, IK 를 맞추려 하면 크래시한다.
            mismatch = check_robot_match(jst, cfg.urdf_path, cfg.dof)
            if mismatch:
                if not dof_warned:
                    dof_warned = True
                    print("\n" + "=" * 74, flush=True)
                    print(f"  ❌ 로봇이 안 맞습니다 — 모터를 움직이지 않습니다", flush=True)
                    print(f"     {mismatch}", flush=True)
                    print(f"     임시 팔(3-DOF)이면  ./run.sh --temp --motors jetson",
                          flush=True)
                    print(f"     전체 팔(5-DOF)이면  ./run.sh --motors jetson", flush=True)
                    print("=" * 74 + "\n", flush=True)
                motor_armed = False
                motors.hold()
                motors.write_positions(motors.read_positions(),
                                       dq=np.zeros(cfg.dof), engaged=False)
                source.publish_state({**build_joint_state(cfg, q, clamped=clamped),
                                      "dof_mismatch": mismatch,
                                      "receiver": receiver,
                                      "motors": motors.summary()})
                time.sleep(1.0 / args.rate)
                continue
            # 연결 절차 3) 실제 관절각으로 IK 시작점을 맞춘다.
            #   이걸 빠뜨리면 여자하는 순간 팔이 홈 자세로 최대 속도로 튄다.
            #   (젯슨도 여자 직후 500ms 는 지령을 무시하는 이중 안전장치를 갖지만,
            #    그건 상대의 방어이지 우리가 안 해도 되는 이유가 아니다)
            if jst is not None and not motor_synced:
                ik.set_q(np.asarray(jst.q, dtype=float)[: cfg.dof])
                home_ee = ik.fk()
                clutch.reset()
                motor_synced = True
                print(f"  ⟳ 실물 관절각으로 동기: "
                      f"{np.round(np.degrees(jst.q), 1)}°", flush=True)
            if motors.address is not None:      # 비컨으로 배운 주소를 반영
                receiver["label"] = (f"{RECEIVER_NAMES.get(args.profile, args.profile)} "
                                     f"{motors.address[0]}")
            # 젯슨이 재무장을 요구하거나(세션 변경·링크 끊김) 트립되면 무장을 푼다.
            #   그러면 아래에서 HOLD 가 나가 관문이 열리고, 사람이 클러치를 다시
            #   잡아야 RUN 이 나간다. 링크가 붙자마자 사람이 팔에서 손 뗀 사이에
            #   움직이는 것을 막는다.
            if jst is not None and (jst.state == STATE_TRIP or jst.await_rearm):
                if motor_armed:
                    why = "트립" if jst.state == STATE_TRIP else f"링크 {jst.link}"
                    print(f"  ⏸ 무장 해제 ({why}) — 클러치를 다시 잡아야 움직입니다",
                          flush=True)
                motor_armed = False
            elif homing:
                motor_armed = True
            elif engaged:
                # ★ 무장하는 순간(비무장 → 무장) 지령을 실물 위치에 다시 맞춘다.
                #
                #   안 맞추면 조종자가 **클러치를 잡았을 뿐인데 팔이 스스로 큰
                #   동작을 시작한다.** 예: 클러치를 안 잡은 채 A(홈 복귀)를 누르면
                #   IK 는 홈으로 점프하는데 실물은 그대로다. 그 상태에서 클러치를
                #   잡으면 그 차이만큼 쓸고 간다 (실측: 40.0°, 시운전 속도로 1.9초).
                #   HOLD 로 오래 있어서 실물이 처졌을 때도 같다.
                #
                #   연결 절차 3단계(실제 q 로 IK 시작점 맞추기)를 기동 때만이
                #   아니라 **무장할 때마다** 적용하는 것이다.
                if not motor_armed and jst is not None and jst.q:
                    q_real = np.asarray(jst.q, dtype=float)[: cfg.dof]
                    drift = float(np.abs(ik.q - q_real).max())
                    ik.set_q(q_real)
                    home_ee = ik.fk()
                    clutch.reset()
                    if drift > np.radians(1.0):
                        print(f"  ⟳ 무장 — 지령이 실물과 {np.degrees(drift):.1f}° "
                              f"벌어져 있어 실물 위치로 다시 맞췄습니다", flush=True)
                motor_armed = True
            if motor_armed:
                motors.enable()      # mode=RUN. 클러치를 떼도 RUN 을 유지하고
            else:                    #   목표만 얼린다 (소자하면 팔이 떨어진다)
                motors.hold()
            # ★ 손은 반드시 **서보각 8개**로 보낸다. 관절각으로 보내면 실물에
            #   없는 자세가 나온다 (30_AGREED A-7). grasp 도 같이 실어서 받는
            #   쪽이 서보를 못 쓰는 경우에 대비한다 — servo 가 우선한다.
            # ★ 지령 속도(dq)를 같이 보낸다. 없으면 받는 쪽이 v_des=0 을 넣게
            #   되고, 그러면 kd 항이 로봇의 자기 움직임에 제동을 건다
            #   (실측 -2.26 N·m, 브레이크어웨이보다 큼).
            #   Mac 은 제어 주기를 알고 있어 정확한 값을 줄 수 있다.
            dq_cmd = (q - q_prev_cmd) * args.rate if q_prev_cmd is not None else None
            q_prev_cmd = q.copy()
            motors.write_positions(
                q, dq=dq_cmd,
                grasp=None if hand is None else float(grasp),
                servo=None if hand is None else grasp_to_servo(grasp),
                # ★ 받는 쪽은 클러치를 모른다. 데이터셋에서 조작자가 손을 뗀
                #   구간을 학습에서 빼려면 이 정보가 필요하다.
                engaged=bool(engaged),
            )

            if homing and jst is not None and jst.q:
                q_real = np.asarray(jst.q, dtype=float)[: cfg.dof]
                if home_reached(q_real, np.asarray(cfg.home, dtype=float)):
                    homing = False
                    motor_armed = False
                    q_prev_cmd = None
                    ik.set_q(cfg.home)
                    home_ee = ik.fk()
                    clutch.reset()
                    motors.hold()
                    motors.write_positions(q_real, dq=np.zeros(cfg.dof), engaged=False)
                    print("  ✅ HOME 정렬 완료 — Grip 을 눌러 조작하세요", flush=True)

        if viz is not None:
            viz.display(ik.robot.state.q)
            robot_frame_viz(ik.robot, cfg.ee_frame)
            frame_viz("target", target, opacity=0.5)

        loop_times.append(time.perf_counter() - t_loop)

        # 브라우저(Quest HUD / 대시보드)로 로봇 상태 발행.
        # 헤드셋을 쓰면 Mac 화면을 못 보므로 관절 한계를 VR 안에서 봐야 한다.
        ee_now = ik.fk()
        reach_now = float(np.linalg.norm(ee_now[:3, 3] - cfg.shoulder))
        # 뷰 변환 = 제어 매핑의 역행렬. 이렇게 해야 "손을 오른쪽으로 옮기면
        # 화면에서도 EE 가 오른쪽으로" 가 미러/yaw 설정과 무관하게 항상 성립한다.
        # ★ 링크마다 적용하지 않고 로봇 전체 그룹에 한 번만 적용한다 (arm_visual.poses 참고).
        view = np.linalg.inv(clutch.mapping)
        state = build_joint_state(cfg, q, clamped=clamped)
        state.update({
            "view_matrix": view_matrix4(view),
            "geometry": visual.geometry,
            "links": visual.poses(),
            "ee_point": visual.point(ee_now[:3, 3]),
            "target_point": visual.point(target[:3, 3]),
            "workspace": {"origin": visual.point(cfg.shoulder),
                          "min": cfg.min_reach, "max": cfg.max_reach},
            "robot": {"name": Path(cfg.urdf_path).name, "dof": cfg.dof,
                      "ee_frame": cfg.ee_frame, "scale": scale,
                      "position_rank": cfg.position_rank},
            # ★ 실물이 붙어 있는데 Isaac 주소로 잘못 들어가면 조종은 되는데
            #   실물이 안 움직인다(반대도 마찬가지). 헤드셋을 쓰면 터미널을
            #   못 보므로 화면에 없으면 알 방법이 없다.
            #   사람이 알고 싶은 건 포트가 아니라 "지금 무엇이 움직이나" 다.
            "receiver": receiver,
            # 여기까지 왔다는 것은 이번 프레임에 컨트롤러 pose 가 들어왔다는 뜻이다.
            "quest": "ok",
            "engaged": bool(engaged),
            "software_stop": False,
            "homing": bool(homing),
            "mapping": {"yaw_deg": float(clutch.base_yaw_deg), "mirror": bool(clutch.mirror),
                        # 미러는 화면을 실물의 거울상으로 만든다. 손처럼 좌우가
                        # 구분되는 부품이 있으면 조작자가 엄지 위치를 반대로 인지한다.
                        "chirality_warning": bool(clutch.mirror and hand is not None)},
            "trigger": float(right.trigger),
            "hand": None if hand is None else {
                "grasp": float(grasp),
                "enabled": bool(grasp_enabled),
                "servo_deg": [round(float(v), 1) for v in np.degrees(grasp_to_servo(grasp))],
                "n_servo": hand.n_servos, "n_joint": hand.n_joints,
            },
            "motors": None if motors is None else motors.summary(),
            # ★ 실제 관절각. 지금까지 화면은 IK 목표만 그려서, 모터가 부하에
            #   걸리거나 속도 제한에 막혀도 **완벽하게 따라오는 것처럼 보였다.**
            #   사람이 그걸 모르고 계속 밀어붙인다. 지령과 실제가 벌어지는 건
            #   이상을 알리는 가장 이른 신호다.
            #   LeRobot 데이터셋도 observation.state(실제) 와 action(지령)을
            #   둘 다 기록하므로 이 통로를 그대로 쓴다.
            "q_act": None if motors is None else
                     [round(float(v), 6) for v in motors.read_positions()],
            "q_cmd": [round(float(v), 6) for v in q],
            "wrist_roll_deg": float(np.degrees(wrist_roll)),
            "ee": {"x": float(ee_now[0, 3]), "y": float(ee_now[1, 3]), "z": float(ee_now[2, 3])},
            "target": {"x": float(target[0, 3]), "y": float(target[1, 3]), "z": float(target[2, 3])},
            "ik_err_mm": float(np.linalg.norm(ee_now[:3, 3] - target[:3, 3]) * 1000),
            "ik_ms": float(np.mean(ik_times[-60:]) * 1000) if ik_times else None,
            "loop_hz": float(1.0 / np.mean(loop_times[-60:])) if loop_times else None,
            "reach": {"current": reach_now, "min": cfg.min_reach, "max": cfg.max_reach,
                      "span_max": cfg.max_reach * 1.15},
        })
        source.publish_state(state)

        now = time.time()
        if now - last_ip_check >= 5.0:
            last_ip_check = now
            current_ip = get_local_ip()
            if current_ip != startup_ip:
                print(f"\n  ⚠️  IP 가 바뀌었습니다: {startup_ip} → {current_ip}", flush=True)
                print(f"      헤드셋에서 열어둔 주소가 더 이상 유효하지 않습니다.", flush=True)
                print(f"      새 주소: https://{current_ip}:{args.port}", flush=True)
                print(f"      인증서를 다시 발급하려면 스크립트를 재시작하세요.\n", flush=True)
                startup_ip = current_ip

        if now - last_print >= 0.5:
            ee = ik.fk()
            err = float(np.linalg.norm(ee[:3, 3] - target[:3, 3]))
            hz = 1.0 / np.mean(loop_times[-60:]) if loop_times else 0.0
            print(
                f"  {'GRIP●' if engaged else 'idle○'} "
                f"target [{target[0,3]:+.3f} {target[1,3]:+.3f} {target[2,3]:+.3f}] "
                f"| ee err {err*1000:5.1f} mm "
                f"| q(deg) [{' '.join(f'{np.degrees(v):+6.1f}' for v in q)}] "
                f"| trig {right.trigger:.2f} grasp {'ON' if grasp_enabled else 'OFF'} "
                f"{grasp:.2f} roll {np.degrees(wrist_roll):+5.1f}° "
                f"| ik {np.mean(ik_times[-60:])*1000:.2f}ms loop {hz:.0f}Hz",
                flush=True,
            )
            last_print = now

        sleep = period - (time.perf_counter() - t_loop)
        if sleep > 0:
            time.sleep(sleep)

    print("\n" + "=" * 74)
    if loop_times:
        print(f"  루프 {len(loop_times)} 회 | 평균 {1/np.mean(loop_times):.1f} Hz "
              f"| IK 평균 {np.mean(ik_times)*1000:.3f} ms (최대 {np.max(ik_times)*1000:.2f})")
    print("  종료")
    print("=" * 74)
    if motors is not None:
        # ★ 소자하지 않는다. 소자하면 팔이 중력으로 떨어진다.
        #   HOLD 로 두고 링크를 끊으면, 젯슨 워치독이 freeze → 3초 뒤 TRIP 으로
        #   안전하게 정지한다. 소자는 사람이 팔을 받친 뒤 명시적으로 해야 한다.
        motors.hold()
        motors.write_positions(motors.read_positions())
        motors.disconnect()
        print("  모터: HOLD 로 두고 링크를 끊었습니다 (소자하지 않음 — 팔이 떨어집니다)")
    source.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
