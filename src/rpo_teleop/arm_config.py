"""URDF 하나로부터 팔 설정을 자동 도출한다.

URDF 를 교체하면 관절 이름·한계·자유도·작업공간·스케일 계수가 전부 달라진다.
그 값들을 스크립트마다 하드코딩해 두면 교체할 때마다 여러 파일을 손봐야 하고,
하나 빠뜨리면 로봇이 엉뚱하게 움직인다. 여기서 URDF 를 단일 진실 공급원으로 삼는다.

    ArmConfig.from_urdf("my_arm.urdf")   # 관절/한계는 파싱, 작업공간은 샘플링 측정
    cfg.save("config/arm.json")          # 측정 결과 캐시 (샘플링은 몇 초 걸린다)
    cfg = ArmConfig.load("config/arm.json")
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# EE 프레임을 자동으로 찾을 때 시도해 볼 이름들 (앞쪽이 우선)
EE_FRAME_CANDIDATES = [
    "ee_link", "ee_frame", "ee",
    "hand_mount", "hand_mount_link", "hand_link", "hand",
    "gripper_frame_link", "gripper_link", "gripper",
    "tool0", "tcp", "tip_link", "tip", "wrist_link",
]

CONTINUOUS_LIMIT = np.pi  # limit 태그가 없는 continuous 관절에 씌울 범위



def safe_reach_bounds(reach: np.ndarray, margin: float) -> tuple[float, float]:
    """샘플링한 리치에서 서로 역전되지 않는 (min, max) 범위를 만든다.

    2-DOF 팔처럼 작업공간이 얇은 구면이면 2-percentile 최소 리치가
    ``raw_max * reach_margin`` 보다 클 수 있다. 그 상태를 그대로 저장하면
    ``np.clip(r, min_reach, max_reach)`` 구간이 거꾸로 되어 모든 목표가 튄다.
    마진을 적용할 수 없을 만큼 얇은 경우에는 실측 최대값을 쓴다.
    """
    reach = np.asarray(reach, dtype=float)
    if reach.size == 0 or not np.all(np.isfinite(reach)):
        raise ValueError("reach must contain finite samples")
    if not 0.0 < margin <= 1.0:
        raise ValueError("reach margin must be in (0, 1]")

    raw_max = float(reach.max())
    raw_min = float(np.percentile(reach, 2))
    max_reach = raw_max * margin
    if max_reach <= raw_min:
        max_reach = raw_max
    min_reach = max(raw_min, max_reach * 0.15)
    min_reach = min(min_reach, max_reach * 0.98)
    return float(min_reach), float(max_reach)


@dataclass
class ArmConfig:
    """한 팔의 기구학 설정. URDF 에서 도출된다."""

    urdf_path: str
    ee_frame: str
    joint_names: list[str]
    lower: list[float]  # rad
    upper: list[float]
    effort: list[float]  # N·m
    velocity: list[float]  # rad/s

    base_frame: str = "base_link"
    shoulder_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """첫 구동 관절의 원점 (base_frame 기준). 리치를 재는 기준점."""

    max_reach: float = 0.5
    min_reach: float = 0.1
    workspace_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    workspace_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    home_q: list[float] = field(default_factory=list)
    """기본 자세. **특이점을 피해서** 고른다.

    관절각을 전부 0으로 두는 게 자연스러워 보이지만, CAD 에서 뽑은 URDF 는 보통
    모든 링크가 일직선인 상태가 0 이라 회전축들이 서로 나란해지는 특이점이 된다.
    실측(robot_arm.urdf): all-zeros 는 자코비안 최소특이값 0.0000, 조건수 5.3e11 이고
    이 자세에서 출발하면 IK 가 관절 한계에 박혀 추종 오차 p95 347 mm 가 났다.
    잘 조건화된 자세(조건수 3.0)로 바꾸면 같은 궤적에서 p95 1.01 mm 로 떨어진다.
    """

    home_condition: float = 0.0
    """홈 자세의 자코비안 조건수. 작을수록 좋다 (10 이하 권장)."""

    total_mass: float = 0.0

    human_reach: float = 0.65
    """성인 어깨~손 리치 기준값 (m). position_scale 산출에 쓴다."""

    reach_margin: float = 0.86
    """측정 최대 리치에 곱할 안전 마진.

    측정 최대치는 팔이 완전히 뻗은 특이 자세라 그 근처에서는 IK 추종이 무너진다.
    robot_arm.urdf 로 마진을 훑어본 결과(연속 궤적 추종 p95):
        0.97 → 44.8 mm   0.94 → 40.2 mm   0.90 → 20.5 mm
        0.86 →  1.5 mm   0.82 →  0.8 mm
    0.86 에서 급격히 떨어지고 그 이상 줄여도 이득이 작아 이 값을 기본으로 둔다.
    """

    control_dt: float = 1.0 / 30.0
    """IK 속도 제한에 쓰는 제어 주기. 제어 루프 주기와 맞춰야 한다."""

    position_rank: int = 3
    """위치 자코비안(3×n)의 랭크. 손끝이 실제로 몇 차원으로 움직일 수 있는지.

    3 이면 정상(부피). 2 면 손끝이 곡면 위에서만 움직인다 — 관절 수와 무관하다.
    임시 3-DOF 팔(link3 자리에 link5)이 그런 경우로, joint3 의 회전축과
    hand_mount 오프셋이 같은 z 축 위에 있어 joint3 이 위치에 전혀 기여하지 않는다
    (±178° 전 범위에서 손끝 이동 0.000 mm 실측).

    3 미만이면 텔레옵에서 '껍질을 따라가는' 움직임만 전달되고 '안팎으로 미는'
    움직임은 무시된다 (실측 전달률: 접선 96.5% / 반경 2.3%). 고장이 아니므로
    시작할 때 안내를 띄운다."""

    # ── 파생값 ────────────────────────────────────────────────────────
    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def position_scale(self) -> float:
        """사람 손 변위 → 로봇 EE 변위 배율."""
        return self.max_reach / self.human_reach

    @property
    def limits(self) -> np.ndarray:
        return np.stack([np.array(self.lower), np.array(self.upper)], axis=1)

    @property
    def shoulder(self) -> np.ndarray:
        return np.array(self.shoulder_pos)

    @property
    def home(self) -> np.ndarray:
        return np.array(self.home_q)

    def clamp_position(self, pos: np.ndarray) -> tuple[np.ndarray, bool]:
        """어깨 기준 리치 구간 안으로 목표 위치를 끌어당긴다."""
        d = np.asarray(pos, dtype=float) - self.shoulder
        r = float(np.linalg.norm(d))
        if r < 1e-9:
            return np.asarray(pos, dtype=float), False
        r_c = float(np.clip(r, self.min_reach, self.max_reach))
        if abs(r_c - r) < 1e-12:
            return np.asarray(pos, dtype=float), False
        return self.shoulder + d * (r_c / r), True

    # ── 저장/불러오기 ─────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ArmConfig:
        return cls(**json.loads(Path(path).read_text()))

    # ── URDF 로부터 생성 ──────────────────────────────────────────────
    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        ee_frame: str | None = None,
        samples: int = 20000,
        seed: int = 0,
    ) -> ArmConfig:
        """URDF 를 읽고 작업공간까지 측정해서 설정을 만든다.

        Args:
            urdf_path: URDF 파일 경로
            ee_frame:  EE 프레임 이름. None 이면 자동 탐색
            samples:   작업공간 측정용 관절각 샘플 수 (0 이면 측정 생략)
        """
        import placo  # 무거우므로 필요할 때만

        urdf_path = Path(urdf_path).resolve()
        robot = placo.RobotWrapper(str(urdf_path))
        joint_names = list(robot.joint_names())
        if not joint_names:
            raise ValueError(f"{urdf_path.name}: 구동 관절이 없습니다.")

        frames = set(robot.frame_names())
        ee = ee_frame or cls._guess_ee_frame(frames, urdf_path)
        if ee not in frames:
            raise ValueError(
                f"EE 프레임 '{ee}' 이 URDF 에 없습니다.\n"
                f"사용 가능한 프레임: {sorted(frames)}"
            )

        lower, upper, effort, velocity, mass = cls._parse_urdf_limits(urdf_path, joint_names)

        robot.update_kinematics()
        # 첫 구동 관절의 프레임 위치를 어깨로 본다 (리치 측정 기준)
        try:
            shoulder = robot.get_T_world_frame(joint_names[0])[:3, 3]
        except Exception:
            shoulder = np.zeros(3)

        base = "base_link" if "base_link" in frames else next(iter(sorted(frames)))
        home = np.clip(np.zeros(len(joint_names)), lower, upper)

        cfg = cls(
            urdf_path=str(urdf_path),
            ee_frame=ee,
            joint_names=joint_names,
            lower=lower.tolist(),
            upper=upper.tolist(),
            effort=effort.tolist(),
            velocity=velocity.tolist(),
            base_frame=base,
            shoulder_pos=shoulder.tolist(),
            home_q=home.tolist(),
            total_mass=float(mass),
        )

        if samples > 0:
            cfg.measure_workspace(robot, samples=samples, seed=seed)
        return cfg

    def measure_workspace(self, robot=None, samples: int = 20000, seed: int = 0) -> None:
        """관절 범위를 균등 샘플링해서 리치와 bounding box 를 측정한다."""
        import placo

        if robot is None:
            robot = placo.RobotWrapper(self.urdf_path)

        rng = np.random.default_rng(seed)
        lo, hi = np.array(self.lower), np.array(self.upper)
        qs = rng.uniform(lo, hi, size=(samples, len(self.joint_names)))

        pts = np.empty((samples, 3))
        for i, q in enumerate(qs):
            for name, val in zip(self.joint_names, q, strict=True):
                robot.set_joint(name, float(val))
            robot.update_kinematics()
            pts[i] = robot.get_T_world_frame(self.ee_frame)[:3, 3]

        reach = np.linalg.norm(pts - self.shoulder, axis=1)
        # 최소 리치: 접힌 자세는 특이점·자기충돌이 몰려 있어 하위 구간을 잘라낸다.
        # 2-DOF 얇은 작업공간에서 reach_margin 때문에 min > max 가 되지 않게
        # 상·하한을 함께 계산한다.
        self.min_reach, self.max_reach = safe_reach_bounds(reach, self.reach_margin)
        self.workspace_min = pts.min(0).tolist()
        self.workspace_max = pts.max(0).tolist()

        self.choose_home(robot, qs, pts)

    def choose_home(self, robot, qs: np.ndarray, pts: np.ndarray) -> None:
        """특이점에서 먼 기본 자세를 고른다.

        자코비안(위치 3행)의 최소 특이값이 클수록 그 자세에서 EE 를 임의 방향으로
        움직이기 쉽다. 작업공간 중간쯤에 있는 자세들 중 이 값이 가장 큰 것을 고른다.
        """
        reach = np.linalg.norm(pts - self.shoulder, axis=1)
        # 사용 가능한 리치 구간 [min_reach, max_reach] 의 가운데쯤에서 고른다.
        # max_reach 만 기준으로 잡으면 홈이 min_reach 에 붙어버려, 팔을 안쪽으로
        # 조금만 당겨도 클램프에 걸린다 (실측: 홈 리치 0.317 vs min 0.254 → 추종 p95 26mm).
        span = self.max_reach - self.min_reach
        band = ((reach > self.min_reach + span * 0.35) &
                (reach < self.min_reach + span * 0.80))
        idx = np.flatnonzero(band)
        if not len(idx):
            idx = np.arange(len(qs))
        # 전수 검사는 비싸므로 후보를 추린다
        rng = np.random.default_rng(0)
        if len(idx) > 400:
            idx = rng.choice(idx, 400, replace=False)

        scored = []
        for i in idx:
            for name, val in zip(self.joint_names, qs[i], strict=True):
                robot.set_joint(name, float(val))
            robot.update_kinematics()
            try:
                jac = robot.frame_jacobian(self.ee_frame, "local_world_aligned")[:3, 6:]
            except Exception:
                continue
            sv = np.linalg.svd(jac, compute_uv=False)
            smin = float(sv.min())
            cond = float(sv.max() / max(smin, 1e-12))
            scored.append((smin, cond, qs[i]))

        if not scored:
            return

        # 조건수만 보고 고르면 관절이 잔뜩 꺾인 기괴한 자세가 뽑힌다. 수치적으로
        # 충분히 좋은(최고의 70% 이상) 후보들 중에서 관절각이 가장 작은 자세를 고른다.
        # 실물 로봇의 기본 자세는 눈으로 봐도 납득이 가야 하고, 자기충돌 위험도 낮다.
        best_smin = max(s for s, _, _ in scored)
        good = [(s, c, q) for s, c, q in scored if s >= 0.7 * best_smin]
        smin, cond, q_best = min(good, key=lambda t: float(np.linalg.norm(t[2])))

        self.home_q = q_best.tolist()
        self.home_condition = cond

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────
    @staticmethod
    def _guess_ee_frame(frames: set[str], urdf_path: Path) -> str:
        for cand in EE_FRAME_CANDIDATES:
            if cand in frames:
                return cand
        # 후보가 없으면 URDF 링크 중 자식이 없는 마지막 링크(말단)를 쓴다
        root = ET.parse(urdf_path).getroot()
        children = {j.find("child").get("link") for j in root.findall("joint")}
        parents = {j.find("parent").get("link") for j in root.findall("joint")}
        leaves = [ln.get("name") for ln in root.findall("link")
                  if ln.get("name") in children and ln.get("name") not in parents]
        if leaves:
            return leaves[-1]
        raise ValueError(
            f"{urdf_path.name}: EE 프레임을 자동으로 찾지 못했습니다. "
            f"--ee-frame 으로 직접 지정하세요. 후보: {sorted(frames)}"
        )

    @staticmethod
    def _parse_urdf_limits(urdf_path: Path, joint_names: list[str]):
        root = ET.parse(urdf_path).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}

        lower, upper, effort, velocity = [], [], [], []
        for name in joint_names:
            j = joints.get(name)
            lim = j.find("limit") if j is not None else None
            jtype = j.get("type") if j is not None else "revolute"
            if lim is not None and lim.get("lower") is not None:
                lower.append(float(lim.get("lower")))
                upper.append(float(lim.get("upper")))
            elif jtype == "continuous":
                lower.append(-CONTINUOUS_LIMIT)
                upper.append(CONTINUOUS_LIMIT)
            else:
                # limit 이 없는 revolute 는 URDF 규격 위반이지만 실무에선 종종 있다
                lower.append(-CONTINUOUS_LIMIT)
                upper.append(CONTINUOUS_LIMIT)
            effort.append(float(lim.get("effort")) if lim is not None and lim.get("effort") else 0.0)
            velocity.append(float(lim.get("velocity")) if lim is not None and lim.get("velocity") else 0.0)

        mass = sum(
            float(m.get("value"))
            for ln in root.findall("link")
            if (m := ln.find("inertial/mass")) is not None
        )
        return (np.array(lower), np.array(upper), np.array(effort),
                np.array(velocity), mass)
