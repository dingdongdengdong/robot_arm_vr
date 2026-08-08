"""팔을 VR 안에서 그리기 위한 형상 추출 + 링크 pose 계산.

설계 원칙: **기구학은 전부 Python 에 둔다.**
브라우저에서 FK 를 다시 구현하면 두 벌이 생기고, 언젠가 반드시 어긋난다.
서버가 placo 로 계산한 링크 world pose 를 그대로 보내고, 브라우저는 받은 자리에
상자를 놓기만 하는 "멍청한 렌더러"로 만든다.

형상은 메시 원본(링크당 12만 정점, 전체 12 MB)을 보내지 않고 **바운딩 박스**로 줄인다.
  · Quest 브라우저가 Wi-Fi 로 12 MB STL 을 받아 파싱하는 건 현실적이지 않다
  · 텔레옵 피드백에는 관절 구조가 보이는 것으로 충분하다
  · 박스 8개면 대역폭이 사실상 0 이라 90 Hz 로도 보낼 수 있다
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def read_stl(path: Path) -> np.ndarray:
    """STL(바이너리/ASCII) → (N,3) 정점. 크기로 형식을 판별한다."""
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + n * 50 and n > 0:
            arr = np.frombuffer(data[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
            tri = arr[:, 12:48].copy().view("<f4").reshape(n, 9)
            return tri.reshape(-1, 3).astype(float)
    verts = [
        [float(x) for x in tok[1:4]]
        for line in data.decode("utf-8", "ignore").splitlines()
        if (tok := line.split()) and len(tok) >= 4 and tok[0] == "vertex"
    ]
    return np.array(verts, dtype=float) if verts else np.zeros((0, 3))


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def extract_link_boxes(urdf_path: str | Path) -> dict[str, dict]:
    """각 링크의 visual 메시 바운딩 박스를 링크 로컬 좌표로 뽑는다.

    Returns:
        {link_name: {"center": [3], "size": [3]}}
        메시가 없는 링크(프레임 표식용)는 결과에서 빠진다.
    """
    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.resolve().parent
    root = ET.parse(urdf_path).getroot()

    boxes: dict[str, dict] = {}
    for link in root.findall("link"):
        chunks = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            f = (urdf_dir / mesh.get("filename")).resolve()
            if not f.exists():
                continue
            v = read_stl(f)
            v = v[np.isfinite(v).all(axis=1)]
            if not len(v):
                continue
            if (scale := mesh.get("scale")):
                v = v * np.array([float(x) for x in scale.split()])
            origin = vis.find("origin")
            if origin is not None:
                rot = rpy_to_matrix([float(x) for x in (origin.get("rpy") or "0 0 0").split()])
                tr = np.array([float(x) for x in (origin.get("xyz") or "0 0 0").split()])
                # numpy 2.x + Apple Accelerate 조합에서 결과가 유한한데도 matmul 중
                # FP 예외 플래그가 올라와 허위 경고가 뜬다. 비유한값은 위에서 이미 걸렀다.
                with np.errstate(all="ignore"):
                    v = v @ rot.T + tr
            chunks.append(v)

        if not chunks:
            continue
        pts = np.vstack(chunks)
        lo, hi = pts.min(0), pts.max(0)
        boxes[link.get("name")] = {
            "center": ((lo + hi) / 2.0).round(5).tolist(),
            "size": np.maximum(hi - lo, 1e-3).round(5).tolist(),
        }
    return boxes


def extract_link_visuals(urdf_path: str | Path, url_prefix: str) -> dict[str, list[dict]]:
    """각 링크의 visual 메시 스펙을 뽑는다 (실제 형상 렌더링용).

    바운딩 박스만으로는 팔이 상자 더미로 보인다. 실제 STL 을 브라우저에서 그리려면
    파일 URL 과 함께 **scale 과 visual origin** 을 그대로 넘겨야 한다. 이 URDF 들은
    메시가 mm/m 로 섞여 있어 scale 을 빠뜨리면 1000배로 나온다.

    Returns:
        {link_name: [{"url", "scale":[3], "xyz":[3], "rpy":[3]}, ...]}
    """
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    out: dict[str, list[dict]] = {}
    for link in root.findall("link"):
        specs = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            fname = Path(mesh.get("filename")).name
            scale = mesh.get("scale")
            origin = vis.find("origin")
            specs.append({
                "url": f"{url_prefix}/{fname}",
                "scale": [float(x) for x in scale.split()] if scale else [1.0, 1.0, 1.0],
                "xyz": [float(x) for x in (origin.get("xyz") if origin is not None
                                           and origin.get("xyz") else "0 0 0").split()],
                "rpy": [float(x) for x in (origin.get("rpy") if origin is not None
                                           and origin.get("rpy") else "0 0 0").split()],
            })
        if specs:
            out[link.get("name")] = specs
    return out


def rotation_to_quat(rot: np.ndarray) -> list[float]:
    """회전행렬 → 쿼터니언 [x, y, z, w] (three.js 순서)."""
    m = np.asarray(rot, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [round(float(v), 6) for v in (x, y, z, w)]


class ArmVisual:
    """링크 형상(정적) + 링크 world pose(매 프레임) 를 만들어 준다."""

    def __init__(self, robot, urdf_path: str | Path, link_names: list[str] | None = None,
                 hand=None, hand_urdf: str | Path | None = None,
                 mount_frame: str = "hand_mount"):
        """
        Args:
            robot: placo.RobotWrapper (제어 루프가 쓰는 것과 같은 인스턴스)
            urdf_path: 메시 바운딩 박스를 뽑을 URDF
            link_names: 그릴 링크. None 이면 메시가 있는 링크 전부
            hand: HandModel (선택). 주면 손 링크도 함께 그린다.
            hand_urdf: 손 URDF 경로. None 이면 hand.urdf_path
            mount_frame: 팔에서 손이 붙는 프레임 이름
        """
        self.robot = robot
        self.hand = hand
        self.mount_frame = mount_frame
        self.hand_boxes = {}
        self.hand_mount_R = None
        self.hand_visuals = {}
        if hand is not None:
            from .hand_model import MOUNT_RPY, MOUNT_WORLD_Y_ROT_RAD, MOUNT_XYZ
            hu = hand_urdf or hand.urdf_path
            self.hand_boxes = extract_link_boxes(hu)
            self.hand_visuals = extract_link_visuals(hu, "/mesh/hand")
            # 사용자가 지정한 Yw +90°는 현재 손 로컬축 회전이 아니라
            # SuperArm HOME의 world/mount Y축 회전이다. 따라서 기본 장착 RPY
            # 앞에 곱한다. 뒤에 곱하면 hand Y축 회전이 되어 결과가 달라진다.
            self.hand_mount_R = (
                rpy_to_matrix((0.0, MOUNT_WORLD_Y_ROT_RAD, 0.0))
                @ rpy_to_matrix(MOUNT_RPY)
            )
            self.hand_mount_t = np.asarray(MOUNT_XYZ, dtype=float)
        self.boxes = extract_link_boxes(urdf_path)
        self.visuals = extract_link_visuals(urdf_path, "/mesh/arm")
        available = set(robot.frame_names())
        # placo 프레임으로 존재하는 것만 남긴다 (메시는 있는데 프레임이 없는 경우 방지)
        self.link_names = [n for n in (link_names or self.boxes.keys()) if n in available]

        # 링크 → 그 링크를 움직이는 관절. 관절이 한계에 걸렸을 때 해당 링크를
        # 색으로 표시하려면 이 대응이 필요하다.
        root = ET.parse(Path(urdf_path)).getroot()
        self.link_joint = {
            j.find("child").get("link"): j.get("name")
            for j in root.findall("joint")
            if j.get("type") not in ("fixed",)
        }

    @property
    def geometry(self) -> list[dict]:
        """정적 형상. 연결 시 한 번만 보내면 된다."""
        out = [
            {
                "name": n,
                "center": self.boxes[n]["center"],
                "size": self.boxes[n]["size"],
                "joint": self.link_joint.get(n),
                "meshes": self.visuals.get(n, []),
            }
            for n in self.link_names
        ]
        if self.hand is not None:
            for n in self.hand.link_names:
                b = self.hand_boxes.get(n)
                if not b:
                    continue
                # 손은 IK 대상이 아니므로 관절 한계 색칠에서 제외한다 (joint=None).
                # 손 관절은 서보로 결정되고, URDF 리밋의 55%는 애초에 도달 불가라
                # "리밋 대비 몇 %" 표시가 오해를 부른다.
                out.append({"name": n, "center": b["center"], "size": b["size"],
                            "joint": None, "part": "hand",
                            "meshes": self.hand_visuals.get(n, [])})
        return out

    def poses(self) -> list[dict]:
        """현재 관절각 기준 각 링크 pose. **로봇 좌표계 그대로** 내보낸다.

        뷰(WebXR) 좌표계 변환은 여기서 하지 않는다. 예전에는 링크마다 V 를 적용했는데
        그러면 링크 로컬 오프셋(박스 중심, 메시 원점)이 함께 변환되지 않아 형상이
        흩어진다. 실제로 그 버그로 팔과 손이 분리돼 보였다.

            링크 로컬 점 c 의 올바른 뷰 좌표  :  V·p + V·R·c
            링크마다 V 를 적용했을 때 렌더 결과:  V·p + (V·R·V⁻¹)·c     ← 다름

        대신 뷰 변환은 **로봇 전체 그룹에 한 번만** 적용한다 (state 의 view_matrix).
        그러면 자식 노드의 로컬 오프셋도 자동으로 따라간다.
        """
        def emit(name, T):
            return {"name": name,
                    "p": [round(float(v), 5) for v in T[:3, 3]],
                    "q": rotation_to_quat(T[:3, :3])}

        out = [emit(n, self.robot.get_T_world_frame(n)) for n in self.link_names]

        if self.hand is not None:
            # 팔의 장착 프레임에 손 밑면을 붙인다.
            T_mount = self.robot.get_T_world_frame(self.mount_frame)
            T_attach = np.eye(4)
            T_attach[:3, :3] = T_mount[:3, :3] @ self.hand_mount_R
            T_attach[:3, 3] = T_mount[:3, 3] + T_mount[:3, :3] @ self.hand_mount_t
            for n in self.hand.link_names:
                if n not in self.hand_boxes:
                    continue
                out.append(emit(n, T_attach @ self.hand.link_pose(n)))
        return out

    def point(self, p_robot) -> list[float]:
        """점 하나를 그대로 내보낸다 (로봇 좌표계). 뷰 변환은 루트 그룹에서 처리."""
        return [round(float(x), 5) for x in np.asarray(p_robot, dtype=float)]
