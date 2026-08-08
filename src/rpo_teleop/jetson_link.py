"""Mac ↔ Jetson Orin Nano UDP 링크.

프로토콜 확정본은 handoff/13_JETSON_REPLY2.txt §1~§7 이다. 이 파일이 그 규격의
Mac 쪽 구현이고, 젯슨 쪽 구현은 jetson_sim.py 가 흉내낸다 (실물이 오면 그 자리에
진짜 젯슨이 들어간다 — 이 파일은 그대로 쓴다).

    Mac                                   Jetson
    ─────────────────────────────────────────────────────────────
    JetsonBackend.write_positions(q)
        └─ UDP 5005 ──── 지령 ──────────> 수신 → 안전계층 → CAN → 모터
    JetsonBackend.read_positions()
        <─ UDP 5006 ──── 상태 ──────────┘ 실제 관절각·온도·에러
        <─ UDP 5007 ──── 비컨 ──────────┘ 1 Hz, 주소 학습용

왜 UDP 인가
    관절 지령은 절대값이라 한 패킷 빠져도 다음 패킷이 덮어쓴다. TCP 는 재전송
    대기 중 뒤 패킷이 막혀서(head-of-line blocking) 오래된 명령이 늦게 도착한다.
    로봇 제어에서는 손실보다 지연이 위험하다.

왜 비컨으로 주소를 학습하는가
    링크가 폰 핫스팟이라 DHCP 임대 갱신 때 양쪽 IP 가 바뀐다. 설정 파일에 IP 를
    박으면 매번 손으로 고쳐야 한다. 젯슨이 브로드캐스트로 자기 주소를 알리고
    Mac 이 그걸 듣는다. 상태 패킷은 젯슨이 '지령 패킷의 송신지'로 되돌려 보내므로
    젯슨은 Mac 의 IP 를 알 필요가 없다.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .motor_backend import MotorBackend

# ── 포트 ──────────────────────────────────────────────────────────────
PORT_CMD = 5005      # Mac → Jetson  지령
PORT_STATE = 5006    # Jetson → Mac  상태
PORT_BEACON = 5007   # Jetson → 브로드캐스트  주소 알림

# ── 지령 모드 (Mac 이 보낸다) ─────────────────────────────────────────
MODE_DISABLED = "DISABLED"   # 토크 없음. 팔이 처진다. 받쳐놓고 쓸 때만.
MODE_HOLD = "HOLD"           # 토크 유지, q 추종 안 함. 링크 확인·일시정지용.
MODE_RUN = "RUN"             # 토크 유지, q 추종.

# ── 젯슨 상태 (젯슨이 보낸다) ─────────────────────────────────────────
STATE_IDLE = "IDLE"          # 소자. 링크는 살아있음.
STATE_HOLD = "HOLD"          # 여자, 자세 유지 중.
STATE_RUN = "RUN"            # 여자, 추종 중.
STATE_TRIP = "TRIP"          # 트립. clear_trip 없이는 안 풀린다.
HOME_TOLERANCE_RAD = np.radians(2.0)


def home_reached(actual: np.ndarray, target: np.ndarray,
                 tolerance: float = HOME_TOLERANCE_RAD) -> bool:
    """모든 활성 관절이 HOME 허용오차 안에 들어왔는지 확인한다."""
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    if actual.shape != target.shape or not np.isfinite(actual).all():
        return False
    return bool(np.abs(actual - target).max(initial=0.0) <= tolerance)


# ──────────────────────────────────────────────────────────────────────
# 패킷
# ──────────────────────────────────────────────────────────────────────
def _nums(vals, digits: int) -> list[float | None]:
    """유한하지 않은 값은 null 로. JSON 에는 NaN/Infinity 가 없다."""
    out = []
    for v in vals:
        f = float(v) if v is not None else None
        out.append(None if f is None or not np.isfinite(f) else round(f, digits))
    return out


@dataclass
class Command:
    """Mac → Jetson (UDP 5005, 30 Hz)."""

    session: int          # Mac 프로세스 시작 시각. 실행 중 불변.
    seq: int              # 단조증가. 젯슨은 역행 패킷을 버린다.
    t: float              # Mac 송신 시각 (로그 대조용. 워치독에 쓰지 말 것)
    mode: str = MODE_HOLD
    q: list[float] = field(default_factory=list)   # 목표 관절각 (rad)
    clear_trip: bool = False
    dq: list[float] | None = None
    """Mac 이 **의도한** 관절속도 (rad/s). 안전계층을 통과하기 **전**의 값이다.

    ★★ 이것을 v_des 에 그대로 넣지 말 것 ★★

      받는 쪽 안전계층이 위치를 클램프하면 실제로 낼 수 있는 속도도 깎인다.
      그런데 v_des 에 깎이기 전 값을 넣으면 kd 항이

          kd × (의도 − 실제) = 2.0 × (3.0 − 1.131) = +3.74 N·m

      만큼 계속 밀어서, **안전계층이 깎은 만큼을 kd 가 되밀어준다.**
      관절이 클램프된 지령보다 kd·Δv/kp 만큼 앞서 달리게 되고, 그만큼
      관절 한계 마진(0.05 rad)을 잡아먹는다.
      → v_des 는 **클램프된 지령**의 차분을 쓸 것 (30_AGREED A-14).

    ★ 그럼 이 값은 어디에 쓰나 — **'사람이 움직이려 하는가' 를 아는 유일한
      신호**다. 받는 쪽의 클램프된 속도는 정지마찰 구간에서 0 이 된다
      (이격 제한이 지령을 고정하므로). 이 값은 클램프 전이라 0 이 아니다.
      마찰 보상을 "의도가 있을 때만" 걸거나, 기록에서 "얼마나 빨리 움직이려
      했는가" 를 남기는 데 쓴다.
    """
    engaged: bool | None = None
    """사람이 클러치(grip)를 잡고 있는가.

    ★ 받는 쪽은 클러치를 모른다. 그런데 데이터셋에서 **조작자가 손을 뗀 구간을
      학습에서 빼려면** 이 정보가 필요하다. 그 구간의 동작은 사람의 의도가
      아니라 "그냥 멈춰 있던 것"이다.
    """
    # 손은 팔과 독립된 SCS0009 직렬 버스로 전달한다. servo 8개가 우선이고,
    # 구형 수신기는 grasp 0..1만 사용해도 된다. None이면 직렬화에서 빠진다.
    grasp: float | None = None
    servo: list[float] | None = None

    def to_bytes(self) -> bytes:
        d = {"session": self.session, "seq": self.seq, "t": round(self.t, 4),
             "mode": self.mode, "q": _nums(self.q, 6),
             "clear_trip": bool(self.clear_trip)}
        if self.dq is not None:
            d["dq"] = _nums(self.dq, 5)
        if self.engaged is not None:
            d["engaged"] = bool(self.engaged)
        if self.grasp is not None:
            d["grasp"] = float(self.grasp)
        if self.servo is not None:
            d["servo"] = _nums(self.servo, 6)
        return json.dumps(d, separators=(",", ":"), allow_nan=False).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Command":
        d = json.loads(raw.decode())
        mode = d.get("mode")
        if mode is None:
            # 하위호환: enable 불리언만 오면 true=RUN, false=HOLD.
            # ★ false 를 DISABLED 로 해석하지 않는다. 위험한 쪽이 기본값이
            #   되면 안 된다 (13_JETSON_REPLY2.txt §1-A).
            mode = MODE_RUN if d.get("enable") else MODE_HOLD
        return cls(session=int(d["session"]), seq=int(d["seq"]), t=float(d.get("t", 0.0)),
                   mode=mode, q=list(d.get("q", [])),
                   clear_trip=bool(d.get("clear_trip", False)),
                   dq=d.get("dq"),
                   engaged=(None if d.get("engaged") is None else bool(d["engaged"])),
                   grasp=d.get("grasp"), servo=d.get("servo"))


@dataclass
class State:
    """Jetson → Mac (UDP 5006, 30 Hz)."""

    session: int | None   # 젯슨이 현재 수락 중인 세션 (에코)
    seq: int              # 마지막으로 처리한 지령의 seq
    t: float
    q: list[float]        # 실제 관절각 (rad)
    dq: list[float] = field(default_factory=list)
    tau: list[float] = field(default_factory=list)
    temp: list[float] = field(default_factory=list)
    err: list[int] = field(default_factory=list)
    state: str = STATE_IDLE
    trip: str | None = None
    rx_age_ms: float = 0.0     # 젯슨이 마지막 지령을 받은 뒤 경과 (monotonic)
    soft_start: float = 0.0    # 소프트스타트 진행률 0~1
    await_rearm: bool = False  # True 면 RUN 을 거부 중. HOLD 를 한 번 보내야 열린다.
    robot: str | None = None
    """받는 쪽이 구동 중인 팔의 식별자 (URDF 파일명, 경로 제외).

    ★ 관절 수 비교만으로는 **관절 수가 우연히 같은 다른 팔**을 못 잡는다.
      둘 다 3축인데 서로 다른 팔이면 조용히 통과하고, 엉뚱한 관절이 움직인다.
      경로는 양쪽이 다르므로 **파일명만** 비교한다.
      값이 없으면(구버전 받는 쪽) 관절 수만 비교한다 — 하위호환.
    """
    link: str = "ok"           # ok | stale | lost — ★ state 와 직교. 이 세 값뿐.
    """통신 건강도. 모터 상태머신(state)과 섞지 않는다.

    ★ 값은 ok / stale / lost 세 개뿐이다. "trip" 을 넣지 말 것.
      트립은 모터 상태이지 링크 상태가 아니다. 워치독 트립 구간의 링크 상태는
      "lost" 가 정확하고, TRIP 여부는 state 로 이미 드러나 정보 손실이 없다.

    state 하나에 통신 상태까지 실으면 대시보드가 "정상 추종 중"으로 보이는
    동안 링크가 죽어 있을 수 있다. 반대로 통신을 state 로 표현하면 모터가
    무슨 상태인지 알 수 없게 된다. 두 축은 독립이어야 한다.
    """

    def to_bytes(self) -> bytes:
        # ★ allow_nan=False. NaN/Infinity 는 JSON 표준에 없다. 파이썬 json 은
        #   기본으로 NaN 을 그냥 써버리고 읽을 때도 받아주지만, 엄격한 파서
        #   (다른 언어/라이브러리)는 패킷 통째로 거부한다. 값이 없으면 null 로.
        #   실제로 이걸 안 하면 대시보드 /state 가 500 으로 죽는다 (겪음).
        return json.dumps({
            "session": self.session, "seq": self.seq, "t": round(self.t, 4),
            "q": _nums(self.q, 6),
            "dq": _nums(self.dq, 4),
            "tau": _nums(self.tau, 3),
            "temp": _nums(self.temp, 1),
            "err": [int(v) for v in self.err],
            "state": self.state, "trip": self.trip,
            "rx_age_ms": round(self.rx_age_ms, 1),
            "soft_start": round(self.soft_start, 3),
            "link": self.link,
            "await_rearm": bool(self.await_rearm),
            "robot": self.robot,
        }, separators=(",", ":"), allow_nan=False).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> "State":
        d = json.loads(raw.decode())
        return cls(session=d.get("session"), seq=int(d.get("seq", 0)),
                   t=float(d.get("t", 0.0)), q=list(d.get("q", [])),
                   dq=list(d.get("dq", [])), tau=list(d.get("tau", [])),
                   temp=list(d.get("temp", [])), err=list(d.get("err", [])),
                   state=d.get("state", STATE_IDLE), trip=d.get("trip"),
                   rx_age_ms=float(d.get("rx_age_ms", 0.0)),
                   soft_start=float(d.get("soft_start", 0.0)),
                   link=d.get("link", "ok"),
                   await_rearm=bool(d.get("await_rearm", False)),
                   robot=d.get("robot"))


@dataclass
class Beacon:
    """Jetson → 브로드캐스트 (UDP 5007, 1 Hz). 주소 학습용."""

    role: str = "jetson"
    ip: str = ""
    port: int = PORT_CMD
    state: str = STATE_IDLE
    session: int | None = None
    t: float = 0.0

    def to_bytes(self) -> bytes:
        return json.dumps({"role": self.role, "ip": self.ip, "port": self.port,
                           "state": self.state, "session": self.session,
                           "t": round(self.t, 3)}, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Beacon":
        d = json.loads(raw.decode())
        return cls(role=d.get("role", ""), ip=d.get("ip", ""),
                   port=int(d.get("port", PORT_CMD)), state=d.get("state", STATE_IDLE),
                   session=d.get("session"), t=float(d.get("t", 0.0)))


def check_robot_match(state: "State | None", urdf_name: str,
                      dof: int) -> str | None:
    """받는 쪽이 이쪽 설정과 같은 팔인가. 다르면 사람이 읽을 설명을 준다.

    ★★ 식별자가 있으면 **식별자만이 권위다.** 관절 수는 식별자가 없을 때의
       폴백이다 (30_AGREED A-10). 둘을 AND 로 걸면 안 된다.

       왜 — 받는 쪽은 팔보다 **많은** 관절을 보낼 수 있다. Isaac 은 팔 5 +
       손 12 = 17개를 보내고, Mac 은 앞의 5개만 쓴다 (doc 18 §2.3, 그리고
       read_positions() 가 q[:n] 으로 자른다). 관절 수를 AND 로 걸면 이
       정상 구성이 **영원히 무장하지 못한다.**
       (실제로 그 상태로 Isaac 갈래가 통째로 막혀 있었다 — 20_ISAAC M-10)

    ★ 왜 검사 자체가 필요한가 — `--temp` 를 빠뜨리면 5-DOF 설정으로 3-DOF
      실물에 지령을 보낸다. 예전에는 여기서 `zip(strict=True)` 가 터져
      **서버가 반쯤 뜬 채 죽었다.** 메시지도 "argument 2 is shorter than
      argument 1" 이라 원인을 알 수 없다.

      죽는 것보다 나쁜 건 **안 죽고 움직이는 것**이다. 관절 수만 보면 관절
      수가 우연히 맞는 다른 팔(예: 둘 다 3축)을 못 잡는다. 그래서 팔 식별자를
      프로토콜에 넣었다.
    """
    if state is None or not state.q:
        return None
    mine = Path(urdf_name).name

    # ★ 경로는 양쪽이 다르므로 파일명만 비교한다.
    if state.robot:
        if Path(state.robot).name != mine:
            return f"받는 쪽은 '{Path(state.robot).name}' 인데 이 설정은 '{mine}' 입니다"
        # 이름이 맞아도 관절이 **모자라면** 막는다. 넘치는 것은 정상(손이 뒤에
        # 붙는다)이지만, 모자란 것은 위험하다 — read_positions() 가 짧은 q 를
        # 0 으로 채우므로, 통과시키면 없는 관절을 "0 도에 있다" 고 믿고 지령한다.
        if len(state.q) < dof:
            return (f"받는 쪽이 관절 {len(state.q)}개만 보냅니다 — "
                    f"이 설정은 {dof}개가 필요합니다 ({mine})")
        return None

    # 식별자가 없는 구버전 — 관절 수로만 판단한다 (하위호환).
    # ★ 여기서는 != 를 유지한다. 식별자가 없으면 "팔+손이라 긴 것" 과 "아예 다른
    #   팔" 을 구별할 방법이 없고, 조용히 엉뚱한 팔을 움직이는 쪽이 더 나쁘다.
    #   길게 보내는 받는 쪽은 robot 필드를 실어 보내면 된다.
    if len(state.q) != dof:
        return (f"받는 쪽은 관절 {len(state.q)}개인데 이 설정은 {dof}-DOF "
                f"({mine}) 입니다 — 받는 쪽이 robot 필드를 안 보내고 있습니다")
    return None


# ──────────────────────────────────────────────────────────────────────
# Mac 쪽 백엔드
# ──────────────────────────────────────────────────────────────────────
class JetsonBackend(MotorBackend):
    """젯슨을 MotorBackend 로 감싼다.

    ★ 안전 제한(관절한계·속도·소프트스타트·이격·워치독)은 여기서 걸지 않는다.
      전부 젯슨이 건다. 이유는 13_JETSON_REPLY2.txt §7 — 상태를 갖는 적분기가
      양쪽에 있으면 두 지령이 서서히 갈라지고 어느 쪽이 진짜인지 알 수 없게
      된다. 권한은 모터에 가까운 쪽 하나여야 한다.
      Mac 은 감시·표시·기록만 한다.
    """

    def __init__(self, n_joints: int = 3, host: str | None = None,
                 cmd_port: int = PORT_CMD, state_port: int = PORT_STATE,
                 beacon_port: int = PORT_BEACON, session: int | None = None,
                 discover: bool = True):
        """
        Args:
            host: 젯슨 주소. None 이면 비컨으로 학습한다 (핫스팟 권장).
            discover: 비컨 수신 여부. host 를 줘도 비컨으로 주소가 바뀌면 따라간다.
        """
        self._n = n_joints
        self._host = host
        self._cmd_port = cmd_port
        self._state_port = state_port
        self._beacon_port = beacon_port
        self._discover = discover
        # 세션은 프로세스 시작 시각. 재시작하면 바뀌므로 젯슨이 seq 추적을
        # 리셋할 수 있다 (안 그러면 seq 가 0 으로 돌아가 영원히 버려진다).
        self.session = int(session if session is not None else time.time())

        self._seq = 0
        self._sock: socket.socket | None = None
        self._state_sock: socket.socket | None = None
        self._beacon_sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._last_state: State | None = None
        self._last_state_at: float | None = None
        self._mode = MODE_HOLD
        self._clear_trip_until = 0.0
        self._q_cmd = np.zeros(n_joints)
        self._q_cmd_at: float | None = None
        self._rx_count = 0
        self._beacon_at: float | None = None

    # ── MotorBackend ─────────────────────────────────────────────────
    @property
    def n_joints(self) -> int:
        return self._n

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._state_sock.bind(("", self._state_port))
        self._state_sock.settimeout(0.2)
        self._spawn(self._rx_state_loop)

        if self._discover:
            self._beacon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._beacon_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._beacon_sock.bind(("", self._beacon_port))
            self._beacon_sock.settimeout(0.2)
            self._spawn(self._rx_beacon_loop)

    def disconnect(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()
        for s in (self._sock, self._state_sock, self._beacon_sock):
            if s is not None:
                s.close()
        self._sock = self._state_sock = self._beacon_sock = None

    def enable(self) -> None:
        """추종 시작 (mode=RUN)."""
        self._mode = MODE_RUN

    def hold(self) -> None:
        """토크는 유지한 채 추종만 정지 (mode=HOLD)."""
        self._mode = MODE_HOLD

    def disable(self) -> None:
        """소자 (mode=DISABLED). ★ 팔이 중력으로 떨어진다."""
        self._mode = MODE_DISABLED

    # 사람이 버튼을 한 번 누른 것을 몇 프레임 동안 재전송할지 (초).
    # ★ 한 프레임만 보내면 그 패킷이 손실될 때 사람의 조작이 조용히 사라진다
    #   (실제로 겪음 — 손실 있는 링크에서 버튼을 눌러도 트립이 안 풀렸다).
    #   그렇다고 계속 보내면 자동 재시도가 되어 "자동 복구 금지"가 무너진다.
    #   그래서 '한 번의 누름'을 짧은 창(窓) 동안만 재전송한다. 창이 지나면
    #   사람이 다시 눌러야 한다.
    CLEAR_TRIP_RETRY_S = 1.0

    def clear_trip(self) -> None:
        """트립 해제 요청. 사람이 버튼을 눌렀을 때만 호출할 것."""
        self._clear_trip_until = time.monotonic() + self.CLEAR_TRIP_RETRY_S

    def read_positions(self) -> np.ndarray:
        with self._lock:
            st = self._last_state
        if st is None or not st.q:
            return self._q_cmd.copy()
        q = np.array([0.0 if x is None else float(x) for x in st.q], dtype=float)
        if q.size < self._n:                      # 젯슨이 짧게 보내면 0 으로 채운다
            q = np.concatenate([q, np.zeros(self._n - q.size)])
        return q[: self._n]

    def write_positions(self, q: np.ndarray, dq: np.ndarray | None = None,
                        grasp: float | None = None,
                        servo: np.ndarray | None = None,
                        engaged: bool | None = None) -> None:
        """지령 전송. 텔레옵 루프에서 매 프레임 부른다.

        Args:
            q:     팔 관절각 (rad)
            dq:    Mac 이 **의도한** 관절속도 (rad/s). 안 주면 직전 지령과의
                   차분으로 구한다. ★ 클램프 전 값이므로 v_des 로 쓰면 안 된다.
                     Command.dq 주석 참고.
            grasp: 손 쥠 정도 0~1. 받는 쪽이 8개로 전개한다.
            servo: 손 서보각 8개 (rad). ★ 이게 있으면 grasp 보다 우선한다.

        ★ 손은 반드시 **서보각**으로 보낸다. 관절각으로 보내면 실물에 없는
          자세가 나온다 (관절 리밋 사각형의 55% 가 도달 불가). 보간도 서보
          공간에서 해야 중간에서 최대 11.44° 어긋나는 걸 피한다. (30_AGREED A-7)

        ★ 기본값이 None 이라 기존 호출은 그대로 돌고, 둘 다 None 이면 직렬화에서
          필드가 빠져 와이어 포맷도 바이트 단위로 같다.
        """
        q = np.asarray(q, dtype=float)
        # dq 를 안 주면 직전 지령과의 차분으로 구한다. Mac 은 제어 주기를 알고
        # 있으므로 이 값이 곧 지령 속도다.
        if dq is None:
            now = time.monotonic()
            if self._q_cmd_at is not None and now > self._q_cmd_at:
                dq = (q - self._q_cmd) / (now - self._q_cmd_at)
            self._q_cmd_at = now
        else:
            self._q_cmd_at = time.monotonic()
        self._q_cmd = q.copy()
        addr = self.address
        if self._sock is None or addr is None:
            return                                 # 아직 젯슨을 못 찾음
        self._seq += 1
        # 창이 열려 있고 아직 트립 상태일 때만 재전송한다. 트립이 풀리면 즉시 멈춘다.
        st = self.state
        want_clear = (time.monotonic() < self._clear_trip_until
                      and (st is None or st.state == STATE_TRIP))
        cmd = Command(session=self.session, seq=self._seq, t=time.time(),
                      mode=self._mode, q=list(q), clear_trip=want_clear,
                      dq=None if dq is None else [float(v) for v in dq],
                      engaged=engaged,
                      grasp=None if grasp is None else float(grasp),
                      servo=None if servo is None else [float(v) for v in servo])
        try:
            self._sock.sendto(cmd.to_bytes(), addr)
        except OSError:
            pass                                   # 링크가 죽어도 루프는 살아야 한다

    def read_velocities(self) -> np.ndarray:
        st = self.state
        if st is None or not st.dq:
            return np.zeros(self._n)
        return np.asarray(st.dq, dtype=float)[: self._n]

    def read_temperatures(self) -> np.ndarray:
        st = self.state
        if st is None or not st.temp:
            return np.full(self._n, np.nan)
        v = np.array([np.nan if x is None else float(x) for x in st.temp], dtype=float)
        return np.concatenate([v, np.full(max(0, self._n - v.size), np.nan)])[: self._n]

    def read_errors(self) -> list[int]:
        st = self.state
        if st is None or not st.err:
            return [0] * self._n
        return list(st.err[: self._n])

    # ── 링크 상태 ────────────────────────────────────────────────────
    @property
    def address(self) -> tuple[str, int] | None:
        if self._host:
            return (self._host, self._cmd_port)
        return None

    @property
    def state(self) -> State | None:
        with self._lock:
            return self._last_state

    @property
    def link_age_s(self) -> float:
        """마지막 상태 패킷 수신 후 경과. 링크 건강도의 핵심 지표."""
        with self._lock:
            at = self._last_state_at
        return float("inf") if at is None else time.monotonic() - at

    @property
    def link_ok(self) -> bool:
        return self.link_age_s < 0.5

    @property
    def rx_count(self) -> int:
        return self._rx_count

    @property
    def mode(self) -> str:
        return self._mode

    def summary(self) -> dict:
        """대시보드/HUD 용 요약."""
        st = self.state
        age = self.link_age_s
        return {
            "addr": f"{self._host}:{self._cmd_port}" if self._host else None,
            "mode": self._mode,
            "state": st.state if st else None,
            "trip": st.trip if st else None,
            "link_ok": self.link_ok,
            "link_age_ms": None if age == float("inf") else round(age * 1000, 1),
            "rx_age_ms": st.rx_age_ms if st else None,
            "soft_start": st.soft_start if st else None,
            "temp": list(st.temp) if st and st.temp else None,
            "err": list(st.err) if st and st.err else None,
            "rx_count": self._rx_count,
            "seq": self._seq,
            # 해제 요청이 아직 재전송 창 안에 있는가. 창이 닫혔는데도 TRIP 이면
            # 요청이 전달되지 못한 것이므로 화면에서 다시 누르라고 알려야 한다.
            "clear_pending": time.monotonic() < self._clear_trip_until,
            "link": st.link if st else None,
            "await_rearm": st.await_rearm if st else None,
            "robot": st.robot if st else None,
        }

    # ── 내부 ─────────────────────────────────────────────────────────
    def _spawn(self, fn) -> None:
        th = threading.Thread(target=fn, daemon=True)
        th.start()
        self._threads.append(th)

    def _rx_state_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw, _ = self._state_sock.recvfrom(4096)
            except (socket.timeout, OSError):
                continue
            try:
                st = State.from_bytes(raw)
            except Exception:
                continue                            # 깨진 패킷은 버린다
            with self._lock:
                self._last_state = st
                self._last_state_at = time.monotonic()
                self._rx_count += 1

    def _rx_beacon_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw, src = self._beacon_sock.recvfrom(2048)
            except (socket.timeout, OSError):
                continue
            try:
                b = Beacon.from_bytes(raw)
            except Exception:
                continue
            if b.role != "jetson":
                continue
            # 비컨이 알린 IP 보다 실제 송신지를 우선한다. NAT/다중 인터페이스에서
            # 비컨 본문의 IP 가 우리가 닿을 수 없는 주소일 수 있다.
            host = src[0] or b.ip
            if host and host != self._host:
                self._host = host
                self._cmd_port = b.port
            self._beacon_at = time.monotonic()
