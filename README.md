# RPO Arm — VR 텔레오퍼레이션 + LeRobot 모방학습

Meta Quest 2 로 5-DOF 로봇 팔 + AmazingHand 를 조종하고, 그 데이터로 모방학습을 시킨다.

---

## 서버 시작 — 이것만 보면 된다

```bash
cd /Users/jjh/Desktop/robot_arm_vr
```

| 하고 싶은 것 | 명령 |
|---|---|
| **그냥 조종해보기** (모터 없이) | `./run.sh --temp` |
| **모터까지 붙여서 연습** (터미널 2개) | ↓ 아래 참고 |
| 전체 5-DOF 팔로 | `./run.sh` |
| Isaac Sim 갈래 | `./run.sh --temp --profile isaac` |

모터까지 붙이려면 터미널 두 개가 필요하다.

```bash
# 터미널 1 — 가짜 젯슨 (실물 젯슨이 오면 이 줄이 없어진다)
.venv/bin/python scripts/07_fake_jetson.py --motor-hz 500

# 터미널 2 — 텔레옵
./run.sh --temp --motors jetson
```

실물 젯슨이 붙으면 터미널 1 을 없애고 주소만 준다.

```bash
./run.sh --temp --motors jetson --jetson-host 192.168.55.1
```

끄기: 각 터미널에서 `Ctrl+C`

터미널에 접속 주소가 출력된다. **IP 는 네트워크가 바뀔 때마다 달라지므로 항상 이 출력을 보고 접속할 것.**

```
Quest 접속  : https://<IP>:4443
대시보드    : https://<IP>:4443/dashboard
```

| 화면 | 주소 | 용도 |
|---|---|---|
| Quest 2 | `https://<IP>:4443` → **[Start XR]** | VR 조종 |
| Mac 브라우저 | `https://<IP>:4443/dashboard` | 3D 뷰 · 관절 한계 · 상태 |

> 인증서 경고가 뜨면 **[고급] → [계속 진행]**. 자체 서명 인증서라 정상이다.
> WebXR 은 HTTPS(secure context)에서만 동작해서 필요하다.

### 프로파일 — 두 갈래를 동시에 띄운다

같은 저장소에서 **실물 젯슨 갈래**와 **Isaac Sim 갈래**가 동시에 돈다.
포트가 겹치면 늦게 뜬 쪽이 죽거나 UDP 패킷이 조용히 절반씩 사라진다.

| 프로파일 | 웹 | UDP 지령/상태/비컨 | 용도 |
|---|---|---|---|
| `jetson` (기본) | 4443 | 5005/5006/5007 | 실물 로봇 |
| `isaac` | 4453 | 5015/5016/5017 | Isaac Sim |
| `test` | 4523 | 5085/5086/5087 | 손으로 돌려볼 때 |

```bash
./run.sh --temp --motors jetson                    # 실물 갈래
./run.sh --temp --motors jetson --profile isaac    # Isaac 갈래 (동시 실행 가능)
```

포트는 [src/rpo_teleop/profiles.py](src/rpo_teleop/profiles.py) 한 곳에서만 정한다.
막혀 있으면 시작할 때 **누가 물고 있는지와 조치 방법까지** 찍고 종료한다.

> ⚠️ 헤드셋은 한 번에 한 WebXR 세션만 쓴다. 두 서버를 동시에 띄워둘 수는 있지만
> Quest 에서는 **주소를 골라서** 한쪽에만 들어간다 (`:4443` 또는 `:4453`).

작업 규약은 [handoff/19_REPO_COEXISTENCE.txt](handoff/19_REPO_COEXISTENCE.txt).

### 공유 보드 — 일 시작 전에 한 번

두 클로드(실물 젯슨 / Isaac Sim)가 같은 저장소에서 각자 일한다.
서로에게 영향 가는 변경과 메시지를 [board/](board/) 에 적어둔다.

```bash
./board/check.sh          # 양쪽 최신 부분만
```

| 파일 | 누가 쓰나 |
|---|---|
| [board/10_JETSON.txt](board/10_JETSON.txt) | 실물 담당 (시뮬은 읽기만) |
| [board/20_ISAAC.txt](board/20_ISAAC.txt) | 시뮬 담당 (실물은 읽기만) |
| [board/30_AGREED.txt](board/30_AGREED.txt) | 양쪽 합의 사항 (append 만) |
| [board/00_README.txt](board/00_README.txt) | 사용 규약 |

### 자주 쓰는 옵션

```bash
./run.sh --temp             # 임시 테스트 팔 (config/arm_temp.json)
./run.sh --profile isaac    # Isaac Sim 쪽 포트 블록
./run.sh --base-yaw 90      # 팔을 놓은 방향에 맞춰 수직축 보정 (0/90/180/270)
./run.sh --scale 0.5        # 손 변위 → 로봇 변위 배율 (기본: 리치에서 자동)
./run.sh --no-hand          # 손 없이 팔만
./run.sh --meshcat          # Meshcat 3D 뷰도 (보통 불필요)
./run.sh --help             # 전체 옵션
```

### 안 열릴 때

1. 터미널의 `Quest 접속:` 줄이 **항상 정답 주소**다 (IP 는 계속 바뀐다)
2. Quest 가 Mac 과 **같은 Wi-Fi** 인지 (Quest 설정 → Wi-Fi)
3. 인증서 경고 → [고급] → [계속 진행]
4. Mac 브라우저로 같은 주소를 먼저 열어보기
   - 열리면 서버는 정상 → Quest 쪽 네트워크 문제
5. 실행 중 IP 가 바뀌면 터미널에 `⚠️ IP 가 바뀌었습니다` 가 뜬다. 재시작할 것.

---

## 조작

### 오른손 — 로봇 조종

| 입력 | 동작 |
|---|---|
| **Grip** (중지) 꾹 | 팔 추종 활성화. 떼면 정지, 다시 잡으면 이어서 (clutch) |
| 위치·자세 | 팔 EE 목표. Grip 누른 순간 기준 상대값 |
| **Trigger** (검지) | 집게(tongs) 엔드이펙터 0~1 (아날로그) |
| **A** (홈 버튼) | AmazingHand 폄/쥠 토글. 검지 트리거와 분리 |
| **B** | 소프트웨어 HOLD |
| **썸스틱 X** | 손목 롤 |

### 왼손 — 화면·매핑 조절

| 입력 | 동작 |
|---|---|
| **썸스틱** | 가상 로봇 위치 이동 (grip 잡고 밀면 높이) |
| **썸스틱 누름** | 위치 리셋 |
| **Y** | yaw +90° — 좌우가 안 맞을 때 이걸로 맞춘다 |
| **X** | 미러 토글 — ⚠️ 아래 주의 |
| **Trigger** | 모터 트립 해제 (TRIP 일 때만. 대시보드에도 같은 버튼) |

> ⚠️ **미러는 쓰지 마세요.** 반사(det=−1)라서 화면이 실물의 거울상이 됩니다.
> 이 손은 오른손(`amazing_hand_right`)인데 미러를 켜면 **왼손으로 보입니다**.
> 엄지 위치와 접근 방향을 반대로 인지하게 되어 데이터 품질이 망가집니다.
> 좌우가 안 맞으면 **yaw**(정상 회전)로 맞추세요.

---

## URDF 를 바꿨을 때

```bash
.venv/bin/python scripts/06_setup_urdf.py --urdf assets/robot_arm/robot_arm.urdf
```

관절 한계·작업공간·홈자세·스케일이 `config/arm.json` 에 다시 계산된다.
이 파일이 **단일 진실 공급원**이라 다른 코드는 손댈 필요 없다. 자세한 내용은
[docs/03_urdf_swap.md](docs/03_urdf_swap.md).

---

## 임시 테스트 팔

실물 5-DOF 팔을 쓰기 전에 짧은 팔로 파이프라인을 검증하는 용도.

```bash
./run.sh --temp
```

| 폴더 | 구성 | 위치랭크 | 상태 |
|---|---|---|---|
| [assets/robot_arm_temp/](assets/robot_arm_temp/) | 3-DOF `link3`자리에 `link5` | 2 | **현재 사용 중** |
| [assets/robot_arm_temp4/](assets/robot_arm_temp4/) | 4-DOF `link4`자리에 `link5` | 3 | 보관 (미사용) |

### 임시 팔로 조종할 때 알아둘 것

`joint3` 의 회전축과 `hand_mount` 오프셋이 같은 z축 위에 있어서 **`joint3` 은
위치를 전혀 못 바꾼다** (±178° 전 범위에서 손끝 이동 0.000 mm). 위치 자유도가
`joint1`, `joint2` 둘뿐이라 손끝이 곡면 위에서만 움직인다.

| 조종 동작 | 전달률 |
|---|---|
| 가리키는 방향 바꾸기 (껍질 따라) | 95% ✅ |
| 안팎으로 밀고 당기기 (반경) | **1%** — 반응 없음 |

**반응이 없는 건 고장이 아니다.** 관절 수가 아니라 축 배치의 문제라 설정으로는
못 고친다. 파이프라인(입력→변환→클러치→IK→지령→시각화→기록) 검증에는 문제없다.

위치 랭크가 3 미만이면 시작 화면과 대시보드에 경고가 뜬다. 랭크는
`06_setup_urdf.py` 가 등록할 때 재서 `config/*.json` 의 `position_rank` 에 넣는다.

배경과 실측은 [handoff/temp_arm/](handoff/temp_arm/) 참고.

---

## 테스트

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## 구조

```
run.sh                  ★ 서버 시작
config/arm.json         URDF 에서 도출된 설정 (단일 진실 공급원)

src/rpo_teleop/
  xr_server.py          WebXR 양방향 서버 (HTTPS + WebSocket)
  xr_source.py          Quest 입력 수신
  transforms.py         WebXR ↔ 로봇 좌표 변환 + 클러치
  arm_config.py         URDF → 관절/한계/작업공간/홈자세 자동 도출
  arm_visual.py         링크 형상·pose (3D 표시용)
  hand_model.py         AmazingHand 서보 ↔ 관절 변환
  motor_backend.py      모터 백엔드 추상화 + 안전 계층 (실물 백엔드는 미구현)
  servo_map/            손 서보 매핑 (기구 담당자 제공)

web/
  index.html            Quest WebXR 페이지
  dashboard.html        Mac 대시보드
  robot_view.js         3D 렌더링 (VR·대시보드 공용)

assets/robot_arm/       팔 URDF + 메시
assets/hand/            손 URDF + 메시

scripts/                단계별 검증 도구 (01~07)
docs/                   기술 문서 · 실측 데이터
handoff/                하드웨어 담당자 인수인계 문서
```

---

## 현재 상태

| 구간 | 상태 |
|---|---|
| Quest 입력 (90 Hz) | ✅ |
| 좌표 변환 + 클러치 | ✅ |
| IK (placo) | ✅ 연속 추종 p95 0.09 mm |
| 안전 계층 | ✅ 소프트스타트·속도제한·3단계 워치독 |
| 3D 표시 (VR + Mac) | ✅ 실제 STL 형상 |
| 손목 롤 방향 | ✅ 사용자 확인 완료 (2026-08-07) |
| 모터 링크 (UDP) | ✅ `--motors jetson` · 가짜 젯슨으로 E2E 검증 |
| **실물 모터** | ⏳ 젯슨에 하드웨어 연결 대기 |
| 카메라 | ⏳ 구성 확정 (본체 후상방 + 손목), 수집 계층 미구현 |
| 데이터 기록 · 학습 | ❌ 미착수 ([docs/06_task_plan.md](docs/06_task_plan.md)) |

### 모터 붙여서 돌리기

```bash
# 터미널 1 — 가짜 젯슨 (실물이 오면 이 줄만 없어진다)
.venv/bin/python scripts/07_fake_jetson.py --motor-hz 500

# 터미널 2
./run.sh --temp --motors jetson
```

실물 젯슨은 `--jetson-host 192.168.55.1` 로 주소만 주면 된다.
프로토콜·상태머신은 [handoff/17_JETSON_REPLY3.txt](handoff/17_JETSON_REPLY3.txt) 참고.

전체 그림은 [docs/05_pipeline_overview.md](docs/05_pipeline_overview.md) 참고.
