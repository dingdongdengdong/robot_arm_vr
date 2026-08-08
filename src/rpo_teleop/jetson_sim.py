"""가짜 젯슨 — 실물 젯슨이 오기 전까지 그 자리를 대신한다.

진짜 젯슨과 **같은 프로토콜·같은 상태머신·같은 안전계층**을 갖고, CAN 대신
MockBackend(1차 지연 가상 모터)를 돌린다. 그래서 이걸로 통과한 Mac 쪽 코드는
실물에서 주소만 바꾸면 그대로 돈다.

이걸 만드는 이유
    · session / mode / clear_trip / 비컨은 전부 문서상 합의일 뿐 아직 아무도
      돌려본 적이 없다. 27 N·m 팔 앞에서 처음 돌려보는 것은 위험하다.
    · 핫스팟 끊김을 실제로 40분 기다렸다 재현할 수 없다. 여기서는 손실률을
      주입해 워치독 3단계를 즉시 확인한다.
    · 실물에서 문제가 나면 이걸로 바꿔본다. 증상이 사라지면 젯슨/모터 쪽,
      남으면 Mac 쪽이다.

실물 젯슨이 이 파일과 달라야 하는 곳은 딱 하나 — _drive() 안에서 MockBackend
대신 CAN 을 때리는 것뿐이다.

    Mac ──UDP 5005 지령──> [seq/session 검사] → [상태머신] → [SafetyLayer] → 모터
        <─UDP 5006 상태──  실제 관절각·온도·에러
        <─UDP 5007 비컨──  1 Hz
"""

from __future__ import annotations

import random
import socket
import threading
import time

import numpy as np

from .jetson_link import (
    MODE_DISABLED,
    MODE_HOLD,
    MODE_RUN,
    PORT_BEACON,
    PORT_CMD,
    PORT_STATE,
    STATE_HOLD,
    STATE_IDLE,
    STATE_RUN,
    STATE_TRIP,
    Beacon,
    Command,
    State,
)
from .motor_backend import MockBackend, SafetyLayer, SafetyLimits


class FakeJetson:
    """젯슨 브리지 시뮬레이터.

    Args:
        n_joints:  프로토콜상 관절 수 (배열 길이). 임시 팔은 3.
        n_motors:  실제로 모터가 붙은 개수. 지금 실물은 2.
                   ★ q[2] 는 버리고 상태에서도 0 으로 채워 보낸다.
                     joint3 은 위치에 기여가 정확히 0 이라 2축만으로도
                     손끝이 갈 수 있는 곳이 3축과 완전히 같다.
        drop_rate: 수신 패킷을 이 확률로 버린다 (핫스팟 흉내). 0~1.
        blackout:  (간격초, 지속초) 주기적으로 통째로 끊는다. 핫스팟 재현용.
    """

    def __init__(self, lower, upper, max_velocity, n_joints: int = 3,
                 n_motors: int = 2, dt: float = 1 / 30, rate_hz: float = 30.0,
                 motor_hz: float | None = None,
                 velocity_scale: float = 0.1, cmd_port: int = PORT_CMD,
                 state_port: int = PORT_STATE, beacon_port: int = PORT_BEACON,
                 host: str = "", drop_rate: float = 0.0,
                 blackout: tuple[float, float] | None = None,
                 tau: float = 0.05, seed: int | None = None,
                 robot: str | None = None):
        self.n_joints = n_joints
        self.n_motors = min(n_motors, n_joints)
        self.robot = robot          # 상태 패킷에 싣는 팔 식별자 (URDF 파일명)
        self.rate_hz = rate_hz
        # ★ 안전계층은 dt 로 한 스텝 이동량을 정한다. 그 계층을 dt 보다 빠르게
        #   돌리면 매 틱마다 dt 만큼 허용되어 **속도 제한이 그 배수만큼 뚫린다.**
        #   실측: dt=1/30 인데 500 Hz 로 돌리면 7.6배 (젯슨에서는 13.5배).
        #   그래서 제어 주기를 dt 에 못박고, 어긋나면 기동 자체를 거부한다.
        self.control_hz = 1.0 / dt
        # 모터 지령 보간 주기. 안전계층 **뒤**에 둔다. 뒤에 두면 보간기가
        # 한계를 넘을 수 없다 (안전계층 출력 사이만 메우므로).
        self.motor_hz = motor_hz
        self.cmd_port, self.state_port, self.beacon_port = cmd_port, state_port, beacon_port
        self.host = host
        self.drop_rate = drop_rate
        self.blackout = blackout
        self._rng = random.Random(seed)

        lo = np.asarray(lower, dtype=float)[: self.n_motors]
        hi = np.asarray(upper, dtype=float)[: self.n_motors]
        vmax = np.asarray(max_velocity, dtype=float)[: self.n_motors]
        self.limits = SafetyLimits(lower=lo, upper=hi, max_velocity=vmax,
                                   velocity_scale=velocity_scale)
        self.safety = SafetyLayer(self.limits, dt=dt)
        self.motors = MockBackend(n=self.n_motors, tau=tau)

        # 상태머신
        self.state = STATE_IDLE
        self.trip: str | None = None
        self.session: int | None = None
        self.last_seq = -1
        self._q_target = np.zeros(self.n_motors)   # 마지막으로 수락한 지령
        self._enabled_at: float | None = None
        self._last_cmd_at: float | None = None     # monotonic. ★ 패킷의 t 를 쓰지 않는다
        self._peer: tuple[str, int] | None = None
        self.last_cmd: Command | None = None      # 마지막으로 수락한 지령 (검사용)

        self._sock: socket.socket | None = None
        self._bsock: socket.socket | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.stats = {"rx": 0, "dropped_sim": 0, "dropped_seq": 0, "tx": 0,
                      "trips": 0, "sessions": 0, "rearm_blocked": 0, "parse_err": 0}
        # ★ 세션이 바뀌거나 링크가 끊긴 뒤에는 RUN 을 거부한다. HOLD/DISABLED
        #   패킷을 한 번 받아야 관문이 열린다. "Mac 이 예의 바르게 굴 것"에
        #   기대지 않고 젯슨이 강제한다 (권한은 모터에 가까운 쪽 하나).
        self._await_rearm = False
        self._link = "ok"
        self._q_safe: np.ndarray | None = None      # 안전계층 출력 (보간기 입력)
        self._actual = np.zeros(self.n_motors)      # 사이클당 한 번만 읽는다

    # ── 수명주기 ─────────────────────────────────────────────────────
    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.cmd_port))
        self._sock.settimeout(0.05)
        self._bsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._bsock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.motors.connect()
        self._spawn(self._rx_loop)
        self._spawn(self._control_loop)      # ★ control_hz — 안전계층
        self._spawn(self._tx_loop)           #    rate_hz   — 상태 송신
        if self.motor_hz:
            self._spawn(self._interp_loop)   #    motor_hz  — 보간 (안전계층 뒤)
        self._spawn(self._beacon_loop)

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()
        for s in (self._sock, self._bsock):
            if s is not None:
                s.close()
        self._sock = self._bsock = None

    # ── 상태머신 ─────────────────────────────────────────────────────
    def _accept(self, cmd: Command) -> None:
        """지령 하나를 받아 상태를 갱신한다. 여기가 규격의 핵심이다."""
        with self._lock:
            # ── 세션 ──────────────────────────────────────────────
            if cmd.session != self.session:
                self.session = cmd.session
                self.last_seq = -1
                self.stats["sessions"] += 1
                # ★ 새 세션은 HOLD 로 받는다. Mac 이 재시작한 직후 팔이 갑자기
                #   움직이지 않게. RUN 은 Mac 이 다시 요청해야 한다.
                # ★ 단 TRIP 중이면 세션이 바뀌어도 TRIP 을 유지한다. 재시작으로
                #   트립이 풀리면 "자동 복구 금지" 원칙이 무너진다.
                if self.state != STATE_TRIP:
                    self.state = STATE_HOLD if self.state != STATE_IDLE else STATE_IDLE
                    # ★ 상태만 HOLD 로 바꾸면 같은 패킷의 mode=RUN 이 바로 아래에서
                    #   다시 RUN 으로 올려버린다. 관문을 세워야 실제로 막힌다.
                    #   위험 시나리오: 사람이 클러치를 잡은 채 Mac 이 죽었다 살아나면
                    #   재시작한 Mac 이 즉시 RUN 을 내보내고 팔이 바로 움직인다.
                    self._await_rearm = True
            elif cmd.seq <= self.last_seq:
                self.stats["dropped_seq"] += 1     # UDP 는 순서가 뒤바뀔 수 있다
                return
            self.last_seq = cmd.seq
            self.last_cmd = cmd
            self._last_cmd_at = time.monotonic()
            # ★ note_command() 로 알린다. safety.clamp() 는 지령이 끊겨도 매 틱
            #   불리므로 clamp 쪽 시계로는 끊김을 절대 감지 못 한다.
            self.safety.note_command()

            # ── 트립 해제 ─────────────────────────────────────────
            if cmd.clear_trip and self.state == STATE_TRIP:
                # DISABLED/HOLD 요청과 함께 온 것만 수락. RUN 중에는 무시.
                if cmd.mode in (MODE_DISABLED, MODE_HOLD):
                    self.trip = None
                    self.safety.clear_trip()
                    self.state = STATE_IDLE        # 바로 RUN 으로 안 간다
                    self.motors.disable()
                    self._enabled_at = None
                return

            if self.state == STATE_TRIP:
                return                              # 트립 중엔 아무것도 안 받는다

            # ── 모드 전이 ─────────────────────────────────────────
            if cmd.mode == MODE_DISABLED:
                if self.state != STATE_IDLE:
                    self.motors.disable()
                    self._enabled_at = None
                self.state = STATE_IDLE
                self._await_rearm = False          # 관문 해제
                return

            if self.state == STATE_IDLE:
                # 여자. 지령은 지금 실제 위치에서 시작한다 (안 그러면 튄다).
                self.motors.enable()
                self._enabled_at = time.monotonic()
                actual = self.motors.read_positions()
                self.safety.reset(actual)
                self._q_target = actual.copy()
                self.state = STATE_HOLD

            if cmd.mode == MODE_HOLD:
                self._await_rearm = False          # 관문 해제
                # RUN → HOLD: 목표를 '마지막 수락 q' 로 둔다.
                # ★ '현재 실제 q' 로 두면 모터가 뒤처진 상태에서 그 뒤처진
                #   위치가 목표로 굳고, 다시 RUN 갈 때 원래 지령으로 튄다.
                self.state = STATE_HOLD
                return

            # ── RUN ───────────────────────────────────────────────
            if self._await_rearm:
                # HOLD/DISABLED 를 한 번 받기 전까지는 RUN 을 거부한다.
                self.stats["rearm_blocked"] += 1
                self.state = STATE_HOLD
                return
            # ★ 여자 직후 500 ms 는 지령을 무시하고 실제 위치를 유지한다.
            #   Mac 이 '실제 q 로 IK 시작점 맞추기'를 빠뜨려도 안 튀게 하는
            #   이중 안전장치.
            if self._enabled_at is not None and (time.monotonic() - self._enabled_at) < 0.5:
                self.state = STATE_RUN
                return
            if cmd.q:
                q = np.asarray(cmd.q, dtype=float)
                self._q_target = q[: self.n_motors].copy()   # q[2] 는 버린다
            self.state = STATE_RUN

    def _tick(self) -> None:
        """제어 1스텝 — ★ 반드시 control_hz(=1/dt) 로만 불러야 한다.

        안전계층은 한 번 불릴 때마다 dt 만큼의 이동을 허용한다. 이걸 dt 보다
        빠르게 부르면 속도 제한이 그 배수만큼 뚫린다. 모터를 더 촘촘히 먹이고
        싶으면 여기가 아니라 _interp_loop 에서 한다 (안전계층 뒤).
        """
        with self._lock:
            stage = self.safety.link_stage() if self._last_cmd_at else "ok"
            # ★ link 는 ok|stale|lost 세 값만 쓴다. trip 은 **모터 상태**이지
            #   링크 상태가 아니다. 링크 필드에 넣으면 범주 오류이고, 실제로
            #   trip 구간의 링크 상태는 "lost" 가 정확하다 (그만큼 지령이 안 옴).
            #   TRIP 여부는 state 로 이미 드러나므로 정보 손실도 없다.
            self._link = "lost" if stage == "trip" else stage

            if self.state in (STATE_HOLD, STATE_RUN):
                if stage == "trip":
                    self.trip = f"워치독: {self.safety.link_age_s:.1f}초 동안 지령 없음"
                    self.safety.trip(self.trip)
                    self.state = STATE_TRIP
                    self.stats["trips"] += 1
                elif stage == "lost" and self.state == STATE_RUN:
                    # ★ 링크가 진짜 끊겼다 → RUN 에서 내려온다. 토크는 유지(HOLD).
                    #   내려오지 않으면 "복구 후 다시 RUN 을 요청" 이라는 절차가
                    #   성립하지 않고 그냥 이어진다. 사람이 그사이 팔에서 손을
                    #   뗐어도 마찬가지다 (실측: 재무장 없이 0.995 rad 이동).
                    #   ※ stale(100ms)에서는 안 내려온다. 한두 프레임 빠졌다고
                    #     매번 클러치를 다시 잡게 하면 조종이 불가능하다.
                    self.state = STATE_HOLD
                    self._await_rearm = True

            if self.state == STATE_IDLE:
                self.motors.read_positions()        # 소자 상태 — 안 움직인다
                self._q_safe = None
                return

            # ★ read_positions() 는 사이클당 한 번만. 실물에서는 CAN 왕복이고,
            #   MockBackend 에서는 이걸 부를 때마다 시뮬이 진행된다.
            actual = self.motors.read_positions()
            self._actual = actual
            self._q_safe = self.safety.clamp(self._q_target, actual)
            if self.motor_hz is None:
                self.motors.write_positions(self._q_safe)

    def _interp_loop(self) -> None:
        """안전계층 출력 사이를 메워 모터에 더 촘촘히 먹인다 (motor_hz).

        ★ 반드시 안전계층 **뒤**에 둔다. 앞이나 안에 두면 한계를 넘는다.
          뒤에 두면 보간기는 이미 제한된 두 점 사이만 지나므로 넘을 수 없다.
        """
        period = 1.0 / self.motor_hz
        prev = None
        while not self._stop.is_set():
            with self._lock:
                target = None if self._q_safe is None else self._q_safe.copy()
            if target is not None:
                if prev is None:
                    prev = target.copy()
                # 안전계층 한 스텝을 motor_hz/control_hz 번에 나눠 접근
                alpha = min(1.0, self.control_hz / self.motor_hz)
                prev = prev + alpha * (target - prev)
                self.motors.write_positions(prev)
            time.sleep(period)

    def _make_state(self) -> State:
        with self._lock:
            # ★ 제어 루프가 이미 읽은 값을 쓴다. 여기서 또 읽으면 실물에서는
            #   CAN 왕복이 2배가 되고, MockBackend 에서는 시뮬이 사이클당
            #   두 번 적분된다.
            actual = self._actual if self._actual is not None else np.zeros(self.n_motors)
            q = np.zeros(self.n_joints)
            q[: self.n_motors] = actual             # 미장착 관절은 0 으로 채운다
            age_ms = 0.0 if self._last_cmd_at is None else \
                (time.monotonic() - self._last_cmd_at) * 1000.0
            if self._enabled_at is None:
                ss = 0.0
            else:
                el = time.monotonic() - self._enabled_at
                ss = min(1.0, el / max(self.limits.soft_start_s, 1e-9))
            # 실물 백엔드는 직전 CAN 피드백의 속도·온도·오류를
            # 반환한다. MockBackend 처럼 온도가 전부 NaN 인 경우만 기존
            # 시뮬레이션 온도를 사용한다. 미장착 관절은 null 이다.
            dq_hw = np.asarray(self.motors.read_velocities(), dtype=float)[:self.n_motors]
            temp_hw = np.asarray(self.motors.read_temperatures(), dtype=float)[:self.n_motors]
            err_hw = list(self.motors.read_errors())[:self.n_motors]
            if temp_hw.size != self.n_motors or not np.isfinite(temp_hw).any():
                temp_hw = np.asarray(
                    [35.0 + 5.0 * abs(float(v)) for v in actual], dtype=float
                )
            dq = [float(v) for v in dq_hw] + [0.0] * (self.n_joints - self.n_motors)
            temp = [float(v) for v in temp_hw] + [None] * (self.n_joints - self.n_motors)
            err = [int(v) for v in err_hw] + [0] * (self.n_joints - self.n_motors)
            return State(
                session=self.session, seq=self.last_seq, t=time.time(),
                q=[float(v) for v in q],
                dq=dq, tau=[0.0] * self.n_joints,
                temp=temp, err=err,
                state=self.state, trip=self.trip,
                rx_age_ms=age_ms, soft_start=ss, link=self._link,
                await_rearm=self._await_rearm, robot=self.robot,
            )

    # ── 루프 ─────────────────────────────────────────────────────────
    def _blacked_out(self) -> bool:
        if not self.blackout:
            return False
        period, dur = self.blackout
        return ((time.monotonic() - self._t0) % period) < dur

    def _spawn(self, fn) -> None:
        th = threading.Thread(target=fn, daemon=True)
        th.start()
        self._threads.append(th)

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw, src = self._sock.recvfrom(8192)
            except (socket.timeout, OSError):
                continue
            self.stats["rx"] += 1
            # 상태는 '지령 패킷의 송신지'로 되돌려 보낸다 → 젯슨은 Mac IP 를
            # 알 필요가 없다. 핫스팟에서 Mac IP 가 바뀌어도 따라간다.
            self._peer = (src[0], self.state_port)
            if self._blacked_out() or (self.drop_rate and self._rng.random() < self.drop_rate):
                self.stats["dropped_sim"] += 1
                continue
            try:
                cmd = Command.from_bytes(raw)
            except Exception:
                self.stats["parse_err"] += 1     # 조용히 버리면 필드명 오타를 못 잡는다
                continue
            self._accept(cmd)

    def _control_loop(self) -> None:
        """★ control_hz(=1/dt) 고정. rate_hz 와 분리한다."""
        period = 1.0 / self.control_hz
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                # CAN 피드백 유실 등으로 제어 스레드만 죽으면 rx/tx 루프는
                # 계속 살아 있어 마치 정상인 것처럼 보인다. 즉시 TRIP
                # 하고 소자해 재가동을 막는다.
                reason = f"제어 피드백 장애: {type(exc).__name__}: {exc}"
                with self._lock:
                    self.trip = reason
                    self.safety.trip(reason)
                    self.state = STATE_TRIP
                    self._await_rearm = True
                    self.stats["trips"] += 1
                    self._q_safe = None
                try:
                    self.motors.disable()
                except Exception:
                    pass
                break
            time.sleep(period)

    def _tx_loop(self) -> None:
        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            peer = self._peer
            if peer and not self._blacked_out():
                try:
                    self._sock.sendto(self._make_state().to_bytes(), peer)
                    self.stats["tx"] += 1
                except OSError:
                    pass
            time.sleep(period)

    def _beacon_loop(self) -> None:
        # ★ 처음 몇 개는 빠르게 보낸다. 1 Hz 로만 보내면 Mac 이 젯슨을 찾는 데
        #   최대 1초가 걸리고, 그 사이 지령이 한 장도 안 나간다. 부팅·재접속
        #   직후가 사람이 "왜 안 붙지" 하고 기다리는 구간이라 여기를 줄인다.
        sent = 0
        while not self._stop.is_set():
            b = Beacon(role="jetson", ip=_local_ip(), port=self.cmd_port,
                       state=self.state, session=self.session, t=time.time())
            if not self._blacked_out():
                for target in ("255.255.255.255", "127.0.0.1"):
                    try:
                        self._bsock.sendto(b.to_bytes(), (target, self.beacon_port))
                    except OSError:
                        pass
                sent += 1
            time.sleep(0.2 if sent < 5 else 1.0)


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
