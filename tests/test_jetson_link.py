"""Mac ↔ Jetson UDP 링크 검증.

실물 젯슨이 오기 전에 프로토콜·상태머신·워치독을 여기서 못박아 둔다. 27 N·m 팔
앞에서 처음 돌려보는 일이 없게 하는 것이 목적이다.

포트는 실제 포트(5005~5007)를 피해 테스트 전용으로 잡는다. 가짜 젯슨이 떠 있는
채로 테스트를 돌려도 서로 간섭하지 않게.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rpo_teleop.jetson_link import (  # noqa: E402
    MODE_DISABLED,
    MODE_HOLD,
    MODE_RUN,
    STATE_HOLD,
    STATE_IDLE,
    STATE_RUN,
    STATE_TRIP,
    Beacon,
    Command,
    JetsonBackend,
    State,
    check_robot_match,
)
from rpo_teleop.jetson_sim import FakeJetson  # noqa: E402

def free_ports(n: int = 3) -> list[int]:
    """OS 에게 비어 있는 포트를 받아온다.

    ★ 고정 번호를 쓰면 안 된다. 이 파일은 저장소와 젯슨 전달 묶음 양쪽에
      있어서 두 pytest 를 동시에 돌리면 같은 포트를 잡고 "Address already in
      use" 로 무더기 에러가 난다.
      PID 로 대역을 나누는 방법도 써봤는데, 테스트 29개가 각각 여러 포트를
      쓰는 바람에 대역이 넘쳐 옆 프로세스를 침범했다. 칸 수를 세는 대신
      **OS 에게 물어보는 쪽**이 확실하다.
    """
    socks, out = [], []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", 0))
        socks.append(s)
        out.append(s.getsockname()[1])
    for s in socks:        # 다 받은 뒤에 닫는다 (같은 포트를 두 번 받지 않게)
        s.close()
    return out
LOWER = np.array([-3.1, -1.74, -3.1])
UPPER = np.array([3.1, 1.74, 3.1])
VMAX = np.array([3.77, 3.77, 3.77])


def wait_until(pred, timeout=3.0, period=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(period)
    return False


class FeedbackFailureBackend:
    """제어 루프에서 CAN 피드백 유실을 재현한다."""

    def __init__(self) -> None:
        self.disable_calls = 0

    def read_positions(self):
        raise ConnectionError("lost raw feedback from motor 1")

    def disable(self):
        self.disable_calls += 1


@pytest.fixture
def link(request):
    """가짜 젯슨 + 백엔드 한 쌍. 테스트마다 포트를 달리해 격리한다."""
    opts = getattr(request, "param", {}) or {}
    cmd, state, beacon = free_ports(3)

    jet = FakeJetson(lower=LOWER, upper=UPPER, max_velocity=VMAX,
                     n_joints=3, n_motors=opts.get("n_motors", 2),
                     cmd_port=cmd, state_port=state, beacon_port=beacon,
                     host="127.0.0.1", velocity_scale=opts.get("velocity_scale", 1.0),
                     drop_rate=opts.get("drop_rate", 0.0), tau=0.01, seed=0)
    jet.limits.soft_start_s = opts.get("soft_start_s", 0.0)
    for k in ("watchdog_freeze_s", "watchdog_linklost_s", "watchdog_s"):
        if k in opts:
            setattr(jet.limits, k, opts[k])
    jet.start()

    be = JetsonBackend(n_joints=3, host="127.0.0.1", cmd_port=cmd,
                       state_port=state, beacon_port=beacon, discover=False)
    be.connect()
    try:
        yield be, jet
    finally:
        be.disconnect()
        jet.stop()


@pytest.fixture
def link_free():
    """FakeJetson 을 옵션을 바꿔가며 여러 번 띄워야 하는 테스트용."""
    made = []
    def make(**opts):
        cmd, st, bc = free_ports(3)
        jet = FakeJetson(lower=LOWER, upper=UPPER, max_velocity=VMAX, n_joints=3,
                         n_motors=2, cmd_port=cmd, state_port=st, beacon_port=bc,
                         host="127.0.0.1", tau=0.001, seed=0, **opts)
        jet.limits.soft_start_s = 0.0
        jet.start()
        be = JetsonBackend(n_joints=3, host="127.0.0.1", cmd_port=cmd,
                           state_port=st, beacon_port=bc, discover=False)
        be.connect()
        made.append((be, jet))
        return jet, be
    yield make
    for be, jet in made:
        try: be.disconnect()
        except Exception: pass
        try: jet.stop()
        except Exception: pass


# ── 패킷 직렬화 ────────────────────────────────────────────────────────
def test_command_roundtrip():
    c = Command(session=1754438400, seq=42, t=1.5, mode=MODE_RUN,
                q=[0.1, -0.2, 0.3], clear_trip=True)
    got = Command.from_bytes(c.to_bytes())
    assert got.session == c.session and got.seq == c.seq and got.mode == c.mode
    assert np.allclose(got.q, c.q) and got.clear_trip is True


def test_command_omits_hand_fields_when_unused():
    """손은 2차. 이번 차수엔 grasp/servo 를 아예 보내지 않는다."""
    raw = Command(session=1, seq=1, t=0.0, q=[0, 0, 0]).to_bytes().decode()
    assert "grasp" not in raw and "servo" not in raw


def test_legacy_enable_false_means_hold_not_disabled():
    """★ enable=false 를 소자로 해석하면 안 된다. 위험한 쪽이 기본값이 되면 안 된다."""
    import json
    raw = json.dumps({"session": 1, "seq": 1, "t": 0.0, "enable": False, "q": [0, 0, 0]}).encode()
    assert Command.from_bytes(raw).mode == MODE_HOLD
    raw = json.dumps({"session": 1, "seq": 1, "t": 0.0, "enable": True, "q": [0, 0, 0]}).encode()
    assert Command.from_bytes(raw).mode == MODE_RUN


def test_state_and_beacon_roundtrip():
    s = State(session=1, seq=7, t=2.0, q=[0.1, 0.2, 0.0], temp=[35.0, 36.0, float("nan")],
              err=[0, 0, 0], state=STATE_RUN, rx_age_ms=33.0, soft_start=0.5)
    got = State.from_bytes(s.to_bytes())
    assert got.seq == 7 and got.state == STATE_RUN and got.rx_age_ms == 33.0
    b = Beacon(role="jetson", ip="10.0.0.5", port=5005, state=STATE_IDLE, t=1.0)
    assert Beacon.from_bytes(b.to_bytes()).ip == "10.0.0.5"


# ── 상태머신 ───────────────────────────────────────────────────────────
def test_control_feedback_failure_trips_and_disables() -> None:
    cmd, state, beacon = free_ports(3)
    jet = FakeJetson(
        lower=LOWER,
        upper=UPPER,
        max_velocity=VMAX,
        n_joints=3,
        n_motors=2,
        cmd_port=cmd,
        state_port=state,
        beacon_port=beacon,
        host="127.0.0.1",
    )
    backend = FeedbackFailureBackend()
    jet.motors = backend

    jet._control_loop()

    assert jet.state == STATE_TRIP
    assert jet._await_rearm is True
    assert "lost raw feedback from motor 1" in jet.trip
    assert jet.stats["trips"] == 1
    assert backend.disable_calls == 1


def test_hold_then_run_tracks_target(link):
    be, jet = link
    be.hold()
    for _ in range(5):
        be.write_positions(np.zeros(3))
        time.sleep(0.02)
    assert wait_until(lambda: jet.state in (STATE_HOLD, STATE_IDLE))

    be.enable()
    target = np.array([0.4, -0.3, 0.2])
    for _ in range(120):
        be.write_positions(target)
        time.sleep(1 / 60)
    assert jet.state == STATE_RUN
    q = jet.motors.read_positions()
    assert np.allclose(q, target[: jet.n_motors], atol=0.05), f"추종 실패: {q}"


def test_third_joint_is_dropped_but_array_stays_length_3(link):
    """모터는 2축뿐이지만 배열 길이는 3 을 유지한다.

    나중에 3번째 모터를 붙일 때 프로토콜을 안 바꾸려는 것. 상태의 q[2] 는 0.
    """
    be, jet = link
    be.enable()
    for _ in range(80):
        be.write_positions(np.array([0.3, -0.2, 1.5]))   # q[2] 에 큰 값
        time.sleep(1 / 60)
    st = be.state
    assert st is not None and len(st.q) == 3
    assert abs(st.q[2]) < 1e-9, f"미장착 관절은 0 이어야 한다: {st.q[2]}"
    assert be.read_positions().shape == (3,)


def test_disabled_stops_motion(link):
    be, jet = link
    be.enable()
    for _ in range(60):
        be.write_positions(np.array([0.5, 0.4, 0.0]))
        time.sleep(1 / 60)
    be.disable()
    for _ in range(10):          # 모드는 패킷에 실려 가야 반영된다
        be.write_positions(np.array([0.5, 0.4, 0.0]))
        time.sleep(0.02)
    assert wait_until(lambda: jet.state == STATE_IDLE)
    frozen = jet.motors.read_positions().copy()
    for _ in range(40):
        be.write_positions(np.array([1.2, 1.2, 0.0]))
        time.sleep(1 / 60)
    assert np.allclose(jet.motors.read_positions(), frozen, atol=1e-6), "소자 상태에서 움직였다"


# ── 세션 ───────────────────────────────────────────────────────────────
def test_seq_rollback_is_rejected(link):
    be, jet = link
    be.enable()
    for _ in range(30):
        be.write_positions(np.array([0.2, 0.1, 0.0]))
        time.sleep(1 / 60)
    before = jet.stats["dropped_seq"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    old = Command(session=be.session, seq=1, t=time.time(), mode=MODE_RUN, q=[9, 9, 9])
    for _ in range(5):
        sock.sendto(old.to_bytes(), ("127.0.0.1", jet.cmd_port))
    sock.close()
    assert wait_until(lambda: jet.stats["dropped_seq"] >= before + 5), "역행 seq 를 안 버렸다"


def test_new_session_resets_seq_and_drops_to_hold(link):
    """★ Mac 재시작 데드락 방지. seq 가 0 으로 돌아가도 통해야 한다."""
    be, jet = link
    arm(be, jet, n=30, q=(0.3, 0.2, 0.0))
    assert jet.state == STATE_RUN
    high_seq = jet.last_seq
    assert high_seq > 5

    be2 = JetsonBackend(n_joints=3, host="127.0.0.1", cmd_port=jet.cmd_port,
                        state_port=free_ports(1)[0], discover=False,
                        session=be.session + 1)
    be2.connect()
    try:
        be2.hold()
        for _ in range(10):
            be2.write_positions(np.zeros(3))       # seq 가 1 부터 다시 시작
            time.sleep(0.02)
        assert jet.last_seq < high_seq, "새 세션인데 seq 를 리셋하지 않았다"
        assert jet.state != STATE_RUN, "새 세션은 RUN 으로 바로 가면 안 된다"
    finally:
        be2.disconnect()


# ── 워치독 3단계 ───────────────────────────────────────────────────────
@pytest.mark.parametrize("link", [{"watchdog_freeze_s": 0.05,
                                   "watchdog_linklost_s": 0.15,
                                   "watchdog_s": 0.4}], indirect=True)
def test_watchdog_freezes_before_tripping(link):
    """0.5초 끊김마다 트립하면 핫스팟에서 운용이 불가능하다.

    freeze 구간에서는 토크를 유지한 채 지령만 얼고, 트립은 '정말 죽었을 때'만.
    """
    be, jet = link
    arm(be, jet, n=60, q=(0.3, -0.2, 0.0))
    assert jet.state == STATE_RUN

    time.sleep(0.08)                                # stale 구간 (0.05~0.15)
    assert jet.state == STATE_RUN, "한두 프레임 빠졌다고 RUN 에서 내려오면 안 된다"
    assert jet.trip is None

    time.sleep(0.20)                                # lost 구간 (0.15~0.4)
    assert wait_until(lambda: jet.state == STATE_HOLD, timeout=0.5), \
        "링크가 진짜 끊기면 RUN 에서 내려와야 한다 (토크는 유지)"
    assert jet.trip is None, "이 구간에서 트립하면 안 된다"

    time.sleep(0.35)                                # 트립 구간 (0.4~)
    assert wait_until(lambda: jet.state == STATE_TRIP, timeout=1.0), "끝내 트립해야 한다"
    assert jet.trip and "워치독" in jet.trip


@pytest.mark.parametrize("link", [{"watchdog_freeze_s": 0.05,
                                   "watchdog_linklost_s": 0.15,
                                   "watchdog_s": 0.3}], indirect=True)
def test_trip_needs_explicit_clear_and_returns_to_idle(link):
    be, jet = link
    arm(be, jet, n=40, q=(0.3, -0.2, 0.0))
    time.sleep(0.5)
    assert wait_until(lambda: jet.state == STATE_TRIP)

    # RUN 을 계속 보내도 안 풀린다
    for _ in range(30):
        be.write_positions(np.array([0.5, 0.3, 0.0]))
        time.sleep(1 / 60)
    assert jet.state == STATE_TRIP, "지령만으로 트립이 풀리면 안 된다"

    # RUN 과 함께 온 clear_trip 은 무시해야 한다
    be.clear_trip()
    be.write_positions(np.array([0.5, 0.3, 0.0]))
    time.sleep(0.1)
    assert jet.state == STATE_TRIP, "RUN 중 clear_trip 은 무시해야 한다"

    # HOLD + clear_trip 이어야 풀린다. 해제 직후 상태는 IDLE(소자).
    be.hold()
    be.clear_trip()
    be.write_positions(np.zeros(3))          # ★ 딱 한 장만 보내고 확인한다
    assert wait_until(lambda: jet.state == STATE_IDLE), "해제 직후에는 IDLE 이어야 한다"
    assert jet.trip is None

    # 그 뒤 HOLD 지령이 계속 오면 여자되어 HOLD 로 올라간다(정상 — 사전 점검 상태).
    # ★ 다만 RUN 으로는 절대 바로 가면 안 된다. Mac 이 RUN 을 다시 요청해야 한다.
    for _ in range(10):
        be.write_positions(np.zeros(3))
        time.sleep(0.02)
    assert jet.state == STATE_HOLD, f"HOLD 지령에는 HOLD 여야 한다: {jet.state}"

    # RUN 을 다시 요청하면 정상 복귀
    be.enable()
    for _ in range(60):
        be.write_positions(np.array([0.2, -0.1, 0.0]))
        time.sleep(1 / 60)
    assert jet.state == STATE_RUN, "해제 후 RUN 요청에는 복귀해야 한다"
    assert np.allclose(jet.motors.read_positions(), [0.2, -0.1], atol=0.05)


@pytest.mark.parametrize("link", [{"watchdog_freeze_s": 0.05,
                                   "watchdog_linklost_s": 0.15,
                                   "watchdog_s": 0.3}], indirect=True)
def test_session_change_does_not_clear_trip(link):
    """Mac 을 재시작하는 것으로 트립이 풀리면 '자동 복구 금지'가 무너진다."""
    be, jet = link
    arm(be, jet, n=40, q=(0.3, -0.2, 0.0))
    time.sleep(0.5)
    assert wait_until(lambda: jet.state == STATE_TRIP)

    be2 = JetsonBackend(n_joints=3, host="127.0.0.1", cmd_port=jet.cmd_port,
                        state_port=free_ports(1)[0], discover=False,
                        session=be.session + 99)
    be2.connect()
    try:
        be2.enable()
        for _ in range(20):
            be2.write_positions(np.array([0.4, 0.3, 0.0]))
            time.sleep(0.02)
        assert jet.state == STATE_TRIP, "세션이 바뀌어도 트립은 유지돼야 한다"
    finally:
        be2.disconnect()


# ── 안전 ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("link", [{"velocity_scale": 0.1, "soft_start_s": 0.0}], indirect=True)
def test_velocity_scale_enforced_on_jetson_side(link):
    """속도 제한 권한은 젯슨에만 있다. Mac 은 안 건다."""
    be, jet = link
    be.enable()
    seen = []
    for _ in range(150):
        be.write_positions(np.array([3.0, 1.7, 0.0]))   # 멀리, 계속
        seen.append(jet.motors.q_cmd.copy())
        time.sleep(1 / 60)
    steps = np.abs(np.diff(np.array(seen), axis=0))
    max_step = float(VMAX[0]) * 0.1 * (1 / 30)
    assert steps.max() <= max_step + 1e-6, f"한 스텝 {steps.max():.5f} > {max_step:.5f}"


@pytest.mark.parametrize("link", [{"soft_start_s": 0.5, "velocity_scale": 1.0}], indirect=True)
def test_soft_start_prevents_jump_on_enable(link):
    be, jet = link
    be.enable()
    first = []
    for _ in range(6):
        be.write_positions(np.array([3.0, 1.7, 0.0]))
        time.sleep(1 / 60)
        first.append(jet.motors.q_cmd.copy())
    assert np.abs(np.array(first)).max() < 0.05, f"여자 직후 튀었다: {np.abs(np.array(first)).max():.4f}"


def test_enable_starts_from_actual_position_not_home(link):
    """여자 직후 500ms 는 지령을 무시하고 실제 위치를 유지한다 (이중 안전장치)."""
    be, jet = link
    jet.motors.set_state(np.array([0.7, -0.5]))
    be.enable()
    moved = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.3:
        be.write_positions(np.array([0.0, 0.0, 0.0]))   # 홈으로 가라는 지령
        moved.append(jet.motors.read_positions().copy())
        time.sleep(1 / 60)
    drift = np.abs(np.array(moved) - np.array([0.7, -0.5])).max()
    assert drift < 0.05, f"실제 위치를 안 지키고 홈으로 움직였다: {drift:.4f}"


# ── 링크 품질 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("link", [{"drop_rate": 0.4}], indirect=True)
def test_survives_packet_loss(link):
    """지령이 절대값이라 40% 손실에도 결국 목표에 도달해야 한다 (UDP 를 고른 이유)."""
    be, jet = link
    target = np.array([0.4, -0.3, 0.0])
    arm(be, jet, n=250, q=tuple(target))
    assert jet.stats["dropped_sim"] > 0, "손실이 주입되지 않았다"
    q = jet.motors.read_positions()
    assert np.allclose(q, target[: jet.n_motors], atol=0.05), f"손실 하에서 추종 실패: {q}"


def test_backend_reports_link_health(link):
    be, jet = link
    be.enable()
    for _ in range(40):
        be.write_positions(np.zeros(3))
        time.sleep(1 / 60)
    assert wait_until(lambda: be.rx_count > 0), "상태 패킷을 못 받았다"
    s = be.summary()
    assert s["link_ok"] is True and s["link_age_ms"] is not None
    assert s["state"] in (STATE_RUN, STATE_HOLD, STATE_IDLE)


def test_beacon_discovery_learns_address():
    """핫스팟에서 IP 가 바뀌어도 설정을 안 고치게 하는 장치."""
    cmd, state, beacon = free_ports(3)
    # ★ 백엔드를 **먼저** 띄운다. 젯슨을 먼저 켜면 connect() 하는 찰나에 비컨이
    #   도착할 수 있고, 그러면 "처음에는 주소를 몰라야 한다" 단언이 무작위로
    #   깨진다 (젯슨 팀이 한 번 겪음). 순서를 고정하면 경쟁 자체가 없어진다.
    be = JetsonBackend(n_joints=3, host=None, cmd_port=cmd, state_port=state,
                       beacon_port=beacon, discover=True)
    be.connect()
    jet = FakeJetson(lower=LOWER, upper=UPPER, max_velocity=VMAX, n_joints=3, n_motors=2,
                     cmd_port=cmd, state_port=state, beacon_port=beacon, host="127.0.0.1")
    try:
        assert be.address is None, "젯슨이 아직 안 떴는데 주소를 알고 있다"
        jet.start()
        assert wait_until(lambda: be.address is not None, timeout=8.0), "비컨으로 주소를 못 배웠다"
        assert be.address[1] == cmd
    finally:
        be.disconnect()
        jet.stop()


def test_write_without_address_is_safe():
    """젯슨을 못 찾은 상태에서 지령을 보내도 예외가 나면 안 된다."""
    p1, p2 = free_ports(2)
    be = JetsonBackend(n_joints=3, host=None, cmd_port=p1,
                       state_port=p2, discover=False)
    be.connect()
    try:
        for _ in range(5):
            be.write_positions(np.array([0.1, 0.2, 0.3]))   # 조용히 버려져야 한다
        assert np.allclose(be.read_positions(), [0.1, 0.2, 0.3])
    finally:
        be.disconnect()


def test_packets_are_strict_json_no_nan():
    """★ NaN 을 실어 보내면 안 된다.

    파이썬 json 은 NaN 을 그냥 써버리고 읽을 때도 받아준다. 그래서 파이썬끼리는
    돌지만 엄격한 파서(다른 언어, 그리고 starlette 의 JSONResponse)는 패킷을
    통째로 거부한다. 실제로 이걸 안 막았을 때 대시보드 /state 가 500 으로 죽었다.
    미장착 관절처럼 값이 없는 자리는 null 로 보낸다.
    """
    import json as _json

    st = State(session=1, seq=1, t=0.0, q=[0.1, 0.2, float("nan")],
               dq=[float("inf"), 0.0, None], tau=[], temp=[35.0, None, float("nan")],
               err=[0, 0, 0], state=STATE_RUN)
    raw = st.to_bytes().decode()
    assert "NaN" not in raw and "Infinity" not in raw, f"NaN/Infinity 가 실렸다: {raw}"

    def strict(c):
        raise ValueError(f"엄격 파서가 거부: {c}")
    d = _json.loads(raw, parse_constant=strict)          # 여기서 터지면 실패
    assert d["q"][2] is None and d["temp"][1] is None

    cmd = Command(session=1, seq=1, t=0.0, q=[0.1, float("nan"), 0.3])
    _json.loads(cmd.to_bytes().decode(), parse_constant=strict)


def test_state_endpoint_survives_missing_motor_values(link):
    """미장착 관절이 있는 상태를 대시보드가 그대로 직렬화할 수 있어야 한다."""
    JSONResponse = pytest.importorskip("starlette.responses").JSONResponse

    be, jet = link
    be.enable()
    for _ in range(40):
        be.write_positions(np.zeros(3))
        time.sleep(1 / 60)
    assert wait_until(lambda: be.state is not None)
    JSONResponse({"motors": be.summary()})               # 500 나던 지점


@pytest.mark.parametrize("link", [{"watchdog_freeze_s": 0.05,
                                   "watchdog_linklost_s": 0.15,
                                   "watchdog_s": 0.3,
                                   "drop_rate": 0.6}], indirect=True)
def test_clear_trip_survives_packet_loss(link):
    """★ 사람이 버튼을 한 번 눌렀는데 그 패킷이 손실되면 조작이 사라진다.

    한 프레임만 보내는 구현에서 실제로 겪었다. 한 번의 누름을 짧은 창 동안만
    재전송해서, 손실이 있어도 전달되고 그렇다고 자동 재시도는 되지 않게 한다.
    """
    be, jet = link
    arm(be, jet, n=40, q=(0.2, -0.1, 0.0))
    time.sleep(0.5)
    assert wait_until(lambda: jet.state == STATE_TRIP), "먼저 트립부터 시켜야 한다"

    be.hold()
    be.clear_trip()                       # 사람이 한 번 누름
    for _ in range(40):                   # 창이 열린 동안 재전송된다
        be.write_positions(np.zeros(3))
        time.sleep(1 / 60)
    assert wait_until(lambda: jet.state != STATE_TRIP, timeout=2.0), \
        "60% 손실에서도 한 번의 누름이 전달돼야 한다"


def test_clear_trip_window_expires(link):
    """창이 지나면 더는 안 보낸다. 계속 보내면 자동 복구가 되어버린다."""
    be, jet = link
    be.clear_trip()
    time.sleep(be.CLEAR_TRIP_RETRY_S + 0.15)
    be.hold()
    be.write_positions(np.zeros(3))
    time.sleep(0.05)
    # 창이 닫힌 뒤 나간 패킷에는 clear_trip 이 실리면 안 된다
    sent = Command(session=be.session, seq=be._seq, t=time.time(), mode=MODE_HOLD,
                   q=[0, 0, 0], clear_trip=time.monotonic() < be._clear_trip_until)
    assert sent.clear_trip is False, "창이 지났는데도 해제 요청이 계속 나간다"


# ── 재무장 관문 (젯슨 팀 지적 2-1 / 2-2) ───────────────────────────────
def arm(be, jet, n=30, q=(0.2, -0.1, 0.0)):
    """연결 절차대로 무장: HOLD 로 관문을 열고 RUN 으로 올린다."""
    be.hold()
    for _ in range(5):
        be.write_positions(np.zeros(3)); time.sleep(0.02)
    be.enable()
    for _ in range(n):
        be.write_positions(np.array(q)); time.sleep(1 / 60)


def test_new_session_refuses_run_until_hold(link):
    """★ 예의 없는 Mac: 재시작 후 첫 패킷부터 RUN 을 쏘는 경우.

    기존 테스트는 be2.hold() 를 먼저 불러서 '예의 바른 Mac' 만 검증했다.
    실제 위험은 사람이 클러치를 잡은 채 Mac 이 죽었다 살아나는 경우다.
    재시작한 Mac 은 클러치가 눌려 있으니 곧바로 RUN 을 내보낸다.
    """
    be, jet = link
    arm(be, jet)
    assert jet.state == STATE_RUN
    q_before = jet.motors.read_positions().copy()

    # 새 세션 — HOLD 없이 RUN 만 쏜다
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(30):
        c = Command(session=be.session + 1, seq=i + 1, t=time.time(),
                    mode=MODE_RUN, q=[1.2, 1.0, 0.0])
        sock.sendto(c.to_bytes(), ("127.0.0.1", jet.cmd_port))
        time.sleep(1 / 60)
    sock.close()

    assert jet.state != STATE_RUN, "새 세션이 HOLD 없이 RUN 으로 올라갔다"
    assert jet.stats["rearm_blocked"] > 0, "관문이 막은 기록이 없다"
    moved = np.abs(jet.motors.read_positions() - q_before).max()
    assert moved < 0.05, f"재무장 없이 {moved:.3f} rad 움직였다"


def test_link_loss_drops_out_of_run_and_requires_rearm(link):
    """★ 링크가 끊겼다 돌아오면 재무장을 요구해야 한다.

    RUN 에서 안 내려오면 "복구 후 다시 RUN 요청" 이라는 절차가 성립하지 않고
    그냥 이어진다. 사람이 그사이 팔에서 손을 뗐어도 마찬가지다.
    """
    be, jet = link
    jet.limits.watchdog_freeze_s = 0.05
    jet.limits.watchdog_linklost_s = 0.15
    jet.limits.watchdog_s = 5.0            # 트립은 안 나게 (재무장만 보려고)
    arm(be, jet)
    assert jet.state == STATE_RUN
    q_before = jet.motors.read_positions().copy()

    time.sleep(0.5)                        # 링크 끊김
    assert wait_until(lambda: jet.state == STATE_HOLD), "lost 인데 RUN 에서 안 내려왔다"
    assert jet.state != STATE_TRIP, "이 구간에서 트립하면 안 된다 (토크는 유지)"

    for _ in range(40):                    # 복구 — RUN 만 계속 보냄
        be.write_positions(np.array([1.2, 1.0, 0.0])); time.sleep(1 / 60)
    moved = np.abs(jet.motors.read_positions() - q_before).max()
    assert moved < 0.05, f"재무장 없이 {moved:.3f} rad 이어갔다"

    arm(be, jet, n=60, q=(1.2, 1.0, 0.0))  # HOLD 로 관문 열고 다시 RUN
    assert jet.state == STATE_RUN, "재무장 후에는 복귀해야 한다"


def test_state_reports_link_separately(link):
    """link(통신)와 state(모터 상태머신)는 직교해야 한다.

    state 하나에 통신 상태까지 실으면 대시보드가 '정상 추종 중'으로 보이는
    동안 링크가 죽어 있을 수 있다.
    """
    be, jet = link
    jet.limits.watchdog_freeze_s = 0.05
    jet.limits.watchdog_linklost_s = 0.15
    jet.limits.watchdog_s = 5.0
    arm(be, jet)
    assert be.state.link == "ok"
    time.sleep(0.4)
    assert wait_until(lambda: be.state and be.state.link in ("stale", "lost")), \
        f"링크가 끊겼는데 link 필드가 {be.state.link}"
    assert be.state.await_rearm is True, "재무장 요구가 상태에 안 보인다"


def test_motor_interpolation_cannot_exceed_velocity_limit(link_free):
    """★ 보간은 안전계층 '뒤' 에 둔다.

    안전계층을 dt 보다 빠르게 돌리면 매 틱 dt 만큼 허용되어 속도 제한이
    그 배수로 뚫린다 (실측: dt=1/30 인데 500Hz 로 돌리면 7.6배).
    보간을 뒤에 두면 이미 제한된 두 점 사이만 지나므로 넘을 수 없다.
    """
    for motor_hz in (None, 500.0):
        jet, be = link_free(velocity_scale=0.1, motor_hz=motor_hz)
        try:
            jet.limits.soft_start_s = 0.0
            arm(be, jet, n=0)
            be.enable()
            q0 = jet.motors.q_cmd.copy()
            t0 = time.time()
            while time.time() - t0 < 1.0:
                be.write_positions(np.array([3.0, 1.7, 0.0])); time.sleep(1 / 30)
            v = np.abs(jet.motors.q_cmd - q0).max() / (time.time() - t0)
            lim = float(VMAX[0]) * 0.1
            assert v <= lim * 1.1, f"motor_hz={motor_hz}: {v:.3f} > {lim:.3f} rad/s"
        finally:
            be.disconnect(); jet.stop()


def test_read_positions_called_once_per_cycle(link):
    """실물에서는 CAN 왕복이고, 목에서는 부를 때마다 시뮬이 진행된다."""
    be, jet = link
    calls = [0]
    orig = jet.motors.read_positions
    def counted():
        calls[0] += 1
        return orig()
    jet.motors.read_positions = counted
    arm(be, jet, n=0)
    be.enable()
    time.sleep(1.0)
    ticks = jet.control_hz * 1.0
    assert calls[0] <= ticks * 1.3, \
        f"사이클당 한 번이어야 한다: {calls[0]}회 / 예상 {ticks:.0f}회"


def test_link_field_never_reports_trip(link):
    """★ link 는 ok|stale|lost 세 값뿐이다.

    trip 은 모터 상태이지 링크 상태가 아니다. 링크 필드에 넣으면 범주 오류이고,
    받는 쪽이 세 값만 처리하도록 짜면 조용히 미분류로 떨어진다.
    (실제로 구현이 문서를 어기고 trip 을 내보내고 있었다 — 젯슨 팀 지적)
    """
    be, jet = link
    jet.limits.watchdog_freeze_s = 0.05
    jet.limits.watchdog_linklost_s = 0.15
    jet.limits.watchdog_s = 0.4
    arm(be, jet, n=20, q=(0.2, -0.1, 0.0))

    seen = set()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.2:          # ok → stale → lost → trip 구간까지
        st = be.state
        if st:
            seen.add(st.link)
        time.sleep(0.02)

    assert seen <= {"ok", "stale", "lost"}, f"정의에 없는 link 값: {seen - {'ok','stale','lost'}}"
    assert "lost" in seen, "링크가 끊겼는데 lost 가 안 나왔다"
    assert jet.state == STATE_TRIP, "이 시점에는 트립돼 있어야 한다 (link 는 lost)"


def test_hold_latches_target_once_on_entry(link):
    """★ HOLD 는 목표를 **진입하는 순간 한 번** 잡는다. 매 프레임 다시 잡으면 안 된다.

    PD 드라이브에서 매 프레임 target 을 현재 q 로 덮으면 (target − q) 가 항상
    0 이라 **복원 토크가 사라진다.** 감쇠항만 남아 중력이 계속 이기고, 팔이
    천천히 주저앉는다. HOLD 의 존재 이유(팔을 떠받치는 것)가 무너진다.
    Isaac 브릿지에서 실제로 이 버그가 나왔다 — 90초에 134.2° 붕괴.
    (board/20_ISAAC.txt M-8)

    여기서는 두 가지를 본다.
      · HOLD 중에 내부 목표가 안 바뀐다
      · HOLD 지령에 실려 온 q 를 무시한다 (RUN 이 아니므로)
    """
    be, jet = link
    arm(be, jet, n=60, q=(0.5, -0.4, 0.0))
    assert jet.state == STATE_RUN
    q_target_at_run = jet._q_target.copy()
    q_actual_at_run = jet.motors.read_positions().copy()

    be.hold()
    for _ in range(90):                      # 3초간 HOLD, 엉뚱한 q 를 실어 보낸다
        be.write_positions(np.array([9.0, 9.0, 9.0]))
        time.sleep(1 / 30)

    assert np.allclose(jet._q_target, q_target_at_run), \
        f"HOLD 중에 목표가 바뀌었다: {q_target_at_run} → {jet._q_target}"
    moved = np.abs(jet.motors.read_positions() - q_actual_at_run).max()
    assert moved < 0.01, f"HOLD 중에 {np.degrees(moved):.2f}° 움직였다"


def test_arming_never_causes_a_sweep(link):
    """★ 무장(클러치 잡기) 순간에 팔이 스스로 큰 동작을 시작하면 안 된다.

    실물이 어딘가에 있는데 조종자가 클러치를 안 잡은 채 A(홈 복귀)를 누르면
    IK 는 홈으로 점프하고 실물은 그대로다. 그 상태에서 클러치를 잡으면
    그 차이만큼 쓸고 간다 (실측 40.0°, 시운전 속도로 1.9초).
    조종자 입장에서는 "클러치를 잡았을 뿐"인데 팔이 혼자 움직인다.

    연결 절차 3단계(실제 q 로 IK 시작점 맞추기)를 기동 때만이 아니라
    **무장할 때마다** 적용해서 막는다.
    """
    be, jet = link
    # 실물이 지령과 다른 자세에 있는 상황을 만든다
    jet.motors.set_state(np.array([0.6, -0.5]))
    arm(be, jet, n=10, q=(0.6, -0.5, 0.0))
    be.hold()
    for _ in range(10):
        be.write_positions(np.array([0.6, -0.5, 0.0]))
        time.sleep(0.02)

    before = jet.motors.read_positions().copy()

    # 무장 직전에 실물 위치로 다시 맞춘다 (05_teleop_sim 이 하는 일)
    q_now = np.asarray(be.state.q, dtype=float)[:2]
    be.enable()
    for _ in range(90):                      # 조종자는 가만히 있다
        be.write_positions(np.array([q_now[0], q_now[1], 0.0]))
        time.sleep(1 / 60)

    moved = np.abs(jet.motors.read_positions() - before).max()
    assert moved < 0.02, f"무장했을 뿐인데 {np.degrees(moved):.2f}° 움직였다"


def test_wrong_robot_with_same_dof_is_caught(link_free):
    """★ 관절 수가 **같은** 다른 팔은 관절 수 비교로 못 잡는다.

    둘 다 3축인데 서로 다른 팔이면 조용히 통과하고 엉뚱한 관절이 움직인다.
    그래서 상태 패킷에 팔 식별자(URDF 파일명)를 넣었다 (30_AGREED A-10).
    경로는 양쪽이 다르므로 파일명만 비교한다.
    """
    jet, be = link_free(robot="a_different_arm.urdf")
    arm(be, jet, n=10, q=(0.0, 0.0, 0.0))
    assert wait_until(lambda: be.state is not None and be.state.robot)
    st = be.state
    assert len(st.q) == 3, "이 시험은 관절 수가 같은 상황을 전제한다"

    why = check_robot_match(st, "/somewhere/robot_arm_temp.urdf", 3)
    assert why is not None, "관절 수가 같은 다른 팔을 통과시켰다"
    assert "a_different_arm.urdf" in why and "robot_arm_temp.urdf" in why


def test_same_robot_different_path_is_accepted(link_free):
    """경로가 달라도 파일명이 같으면 통과해야 한다.

    Mac 은 assets/robot_arm_temp/... , 젯슨은 for_jetson/assets/... 를 쓴다.
    """
    jet, be = link_free(robot="robot_arm_temp.urdf")
    arm(be, jet, n=10, q=(0.0, 0.0, 0.0))
    assert wait_until(lambda: be.state is not None and be.state.robot)
    assert check_robot_match(
        be.state, "/Users/me/assets/robot_arm_temp/robot_arm_temp.urdf", 3) is None, \
        "같은 팔인데 거부했다"


def _state(q, robot=None):
    """검사 함수만 보는 단위 시험용 상태 패킷."""
    n = len(q)
    return State(session=1, seq=1, t=0.0, q=list(q), dq=[0.0] * n, tau=[0.0] * n,
                 temp=[0.0] * n, err=[0] * n, state="HOLD", trip=False,
                 rx_age_ms=5, soft_start=False, await_rearm=False,
                 link="ok", robot=robot)


def test_extra_joints_are_accepted_when_identifier_matches():
    """★ 받는 쪽이 팔보다 **많은** 관절을 보내는 것은 정상이다.

    Isaac 은 팔 5 + 손 12 = 17개를 보내고 Mac 은 앞의 5개만 쓴다 (doc 18 §2.3).
    식별자와 관절 수를 AND 로 걸면 이 정상 구성이 영원히 무장하지 못한다.
    실제로 그 상태로 Isaac 갈래가 통째로 막혀 있었다 (20_ISAAC M-10).
    """
    st = _state([0.0] * 17, robot="robot_arm.urdf")
    assert check_robot_match(st, "/x/assets/robot_arm/robot_arm.urdf", 5) is None, \
        "팔+손 17관절을 5-DOF 설정이 거부했다 — Isaac 갈래가 막힌다"


def test_missing_joints_are_rejected_even_when_identifier_matches():
    """★ 넘치는 것은 정상이지만 **모자란 것**은 막아야 한다.

    read_positions() 가 짧은 q 를 0 으로 채우므로, 통과시키면 없는 관절을
    "0 도에 있다" 고 믿고 지령한다.
    """
    st = _state([0.0] * 3, robot="robot_arm.urdf")
    why = check_robot_match(st, "/x/assets/robot_arm/robot_arm.urdf", 5)
    assert why is not None, "관절이 모자란데 통과시켰다"
    assert "3" in why and "5" in why


def test_identifier_beats_joint_count_in_both_directions():
    """식별자가 있으면 관절 수가 맞아도 이름이 다르면 막는다."""
    st = _state([0.0] * 3, robot="robot_arm_temp.urdf")
    assert check_robot_match(st, "/x/assets/robot_arm/robot_arm.urdf", 3) is not None


def test_legacy_receiver_without_identifier_still_uses_joint_count():
    """★ 식별자가 없으면 != 를 유지한다 (하위호환 + 안전).

    식별자가 없으면 "팔+손이라 긴 것" 과 "아예 다른 팔" 을 구별할 수 없다.
    조용히 엉뚱한 팔을 움직이는 쪽이 더 나쁘므로 막는다.
    """
    assert check_robot_match(_state([0.0] * 3), "/x/robot_arm_temp.urdf", 3) is None
    why = check_robot_match(_state([0.0] * 17), "/x/robot_arm.urdf", 5)
    assert why is not None
    assert "robot" in why, "받는 쪽이 무엇을 하면 되는지 알려줘야 한다"


def test_command_carries_velocity_and_engaged(link):
    """★ dq 가 없으면 받는 쪽이 v_des=0 을 넣어 자기 움직임에 제동을 건다.
       engaged 는 학습에서 '사람이 손 뗀 구간'을 빼는 데 쓴다.
    """
    be, jet = link
    arm(be, jet, n=5, q=(0.0, 0.0, 0.0))
    be.enable()
    for i in range(30):
        be.write_positions(np.array([0.1 * i * 0.05, 0.0, 0.0]),
                           dq=np.array([0.5, 0.0, 0.0]), engaged=True)
        time.sleep(1 / 60)
    assert jet.last_cmd is not None
    assert jet.last_cmd.dq is not None and abs(jet.last_cmd.dq[0] - 0.5) < 1e-6
    assert jet.last_cmd.engaged is True

    # 안 주면 필드가 빠져야 한다 (와이어 포맷 하위호환)
    raw = Command(session=1, seq=1, t=0.0, q=[0.0]).to_bytes().decode()
    assert "dq" not in raw and "engaged" not in raw
