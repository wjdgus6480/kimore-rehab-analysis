# KIMORE 재활운동 수행 정량화 프로젝트 — REQUIREMENTS

> 상태: Draft v0.2 (2026-11-08)
> 단계: Gate 1 진행 중 — 현재 로컬 사본(불완전)으로 구조화·인벤토리 수행. 공식 원본 재확보 전
> 우선순위: [필수] [중요] [선택] [후순위]
> 근거 표기: **[S]** 원저자 Script에서 직접 확인 · **[D]** 데이터 구조·값에서 직접 확인 · **[E]** 변수명·해석 기반 추정
> **[E] 항목은 확정 요구사항의 근거로 쓰지 않는다.**
> 모든 수치는 현재 로컬 사본 기준이며, 원본 재확보 후 `kimore.qc`로 다시 집계한다.

---

## 1. 프로젝트 개요

### 1.1 목적
KIMORE 데이터셋(Kinect v2 기반 재활운동 기록 + 임상평가 점수)을 이용하여
재활운동 수행에서 나타나는 움직임 특성을 정량화하고,
그 정량 지표와 임상평가 점수의 관계 및 예측 가능성을 검증한다.

### 1.2 성격
- 질병을 판별하거나 진단하는 시스템이 아니다.
- 문서·코드·커밋·API 응답에 "진단", "질병 진단 모델" 등의 표현을 쓰지 않는다. [필수]

### 1.3 핵심 원칙 [필수]
1. 원본 데이터는 수정하지 않는다.
2. 불완전한 현재 데이터로 최종 모델을 만들지 않는다.
3. 데이터 확보 전 최종 feature와 모델을 확정하지 않는다.
4. Subject 단위로 leakage를 차단한다.
5. Group이나 Subject ID를 feature로 사용하지 않는다.
6. Age confounding을 항상 검토한다.
7. 작은 표본에서 feature와 모델을 과도하게 늘리지 않는다.
8. 분석을 반복한 뒤 가장 좋은 결과만 선택하지 않는다.
9. 주 분석과 탐색 분석을 사전에 구분한다.
10. 성능이 낮으면 낮은 결과 자체를 기록하고 원인을 분석한다.
11. 모델보다 데이터 품질과 검증 설계를 우선한다.
12. ML 결과가 좋더라도 임상적 진단이나 실제 의료적 효능을 주장하지 않는다.
13. DB/API/Frontend는 ML 검증 이후에만 확장한다.

---

## 2. 연구 목적 및 연구 질문

- **RQ1 (관련성)**: 운동별로 추출한 운동학적 feature는 해당 운동의 임상평가 점수(PO/CF)와 어떤 관련을 보이는가?
- **RQ2 (예측 가능성)**: 운동학적 feature로 PO/CF 점수를 단순 기준선(전체 평균, Group 평균, Age-only)보다 더 정확하게 추정할 수 있는가?
- **RQ3 (집단 간 차이)**: 건강군과 환자군 사이에 운동학적 feature의 차이가 관찰되는가? 그 차이는 연령을 고려한 뒤에도 유지되는가?

### 2.1 결과 해석 원칙 [필수]
모델 성능이 낮은 것은 실패가 아니라 결과로 보고한다. 다음 관점에서 원인을 함께 검토한다.
데이터 품질 · 운동 특성 · 임상점수와 feature의 관계(천장효과 포함) · 연령 교란 · 표본 수 · 일반화 가능성

---

## 3. 데이터 정의

### 3.1 구조 [D]
```
data/raw/
├── CG/   Expert, NotExpert                 # 건강군
└── GPP/  BackPain, Parkinson, Stroke       # 환자군
     └── <SubjectID>/Es1~Es5/{Raw, Label, Script}
```

| Group | 대상자 폴더 수 [D] |
|---|---|
| Expert | 17 |
| NotExpert | 27 |
| BackPain | 8 |
| Parkinson | 16 |
| Stroke | 10 |
| **합계** | **78** |

- 운동 Es1~Es5의 실제 동작 정의: **원본 문서와 원저자 Script로 확인 필요** (현재 미확인)

### 3.2 파일 종류
| 파일 | 내용 | 근거 |
|---|---|---|
| `JointPosition*.csv` | 헤더 없음, 앞쪽 빈 행, 행 끝 쉼표 → 101필드 중 100열 유효 = 25 joints × (x, y, z, tracking state) | [D] |
| `JointOrientation*.csv` | 같은 형식, 100열 = 25 joints × 4값 | [D] |
| 4값의 성분 순서(w, x, y, z 등) | 확인 필요 | — |
| `TimeStamp*.csv` | 1행 1값 + 행 끝 쉼표. 확인한 파일에서 연속값 차이 약 330,000 | [D] |
| Timestamp 단위 | **확인 필요 (추정하지 않음)** | — |
| `ClinicalAssessment_<ID>.xlsx` | 1행: Subject ID + `clinical TS/PO/CF Ex#1~5` 15열 | [D] |
| `SuppInfo_<ID>.xlsx` | Subject ID, Group, Age, Gender | [D] |
| `preproc_ExN.m`, `feat_extract_ExN.m`, `main_ExN.m`, `filtering.m` | 원저자 MATLAB 코드 | [D] 파일 존재 |
| `*.avi`, `*.mp4`, `*.lnk` | 영상·바로가기 — 분석과 저장소에서 제외 | [D] 파일 존재 |

### 3.3 Joint schema
원저자 `preproc_Ex1.m`이 정의한 열 인덱스 [S]: `index_<Joint> = 1, 5, 9, …, 97` (MATLAB 1-based, joint당 4열)

| idx (0-based) | Joint (Script 변수명) [S] | idx | Joint | idx | Joint |
|---|---|---|---|---|---|
| 0 | Spine_Base | 9 | Elbow_Right | 18 | Ankle_Right |
| 1 | Spine_Mid | 10 | Wrist_Right | 19 | Foot_Right† |
| 2 | Neck | 11 | Hand_Right | 20 | Spine_Shoulder |
| 3 | Head† | 12 | Hip_Left | 21 | Tip_Left† |
| 4 | Shoulder_Left | 13 | Knee_Left | 22 | Thumb_Left† |
| 5 | Elbow_Left | 14 | Ankle_Left | 23 | Tip_Right† |
| 6 | Wrist_Left | 15 | Foot_Left† | 24 | Thumb_Right† |
| 7 | Hand_Left | 16 | Hip_Right | | |
| 8 | Shoulder_Right | 17 | Knee_Right | | |

† Script 주석 `% no orientation` [S]
- 이 순서는 Kinect v2 SDK JointType 순서와 일치하는 것으로 보인다 [E]. 원본 재확보 후 재검증한다.
- Position 열 = `idx*4 + {0:x, 1:y, 2:z, 3:tracking state}` [S][D]
- 코드 구현: `src/kimore/io.py`의 `JOINTS`, `position_columns()`, `orientation_columns()`

### 3.4 용어 [필수]
| 용어 | 정의 |
|---|---|
| Subject | 대상자 1명 |
| Exercise | Es1~Es5 |
| **Session** | **Subject 1명 × Exercise 1개** = ML 기본 표본 |
| Repetition | Session 내부의 반복 1회 (내부 단위) |
| Frame | Kinect 1프레임 (내부 단위) |

---

## 4. 데이터 품질 및 Label QC

### 4.1 현재 로컬 사본 상태 [D]
- 세션 폴더 329개 중 Position + Timestamp가 모두 있는 세션: 39개
- Position + Orientation + Timestamp가 모두 있는 세션: 14개
- 내용이 같은 복사본 파일 존재: `ClinicalAssessment_E_ID15(1)`, `ClinicalAssessment_P_ID3(1)`, `SuppInfo_NE_ID24(1)`, `*(1).m` 등
- 다른 대상자 폴더에 들어 있는 파일: `E_ID3` 폴더 안에 `ClinicalAssessment_E_ID1.xlsx`, `SuppInfo_E_ID1.xlsx`
- 형식이 깨진 원본: `S_ID6/Es5`의 Position·Timestamp 파일에서 18번째 데이터 행에 두 행이 붙어 있음(열 수 2배) → 파서가 탐지해 제외
- 원본 파일이 `(1)` 이름으로만 남은 경우가 있음(예: `E_ID14/Es4`, `E_ID15/Es4`) → 원본이 없으면 그 파일을 사용
- 한 운동 폴더에 녹화가 여러 개인 Session이 있음(`NE_ID4/Es1`, `NE_ID20/Es4`, `NE_ID27/Es2`) → 자동 선택하지 않고 warning
- SuppInfo의 Group 값 표기가 파일마다 다름(`E` / `CG_E`)
- `Script/Script/` 중첩 폴더 존재 (원저자 Script 분석: `docs/original_scripts.md`)
- 최신 집계는 `docs/inventory_report.md` (`python -m kimore.qc`가 생성)
- 결론: 현재 사본으로는 최종 분석을 할 수 없다. 공식 원본을 재확보한다. [필수]
- 공식 배포 페이지 안내: 전체 다운로드에 문제가 있으면 하위 폴더(Expert, NotExpert, BackPain, Parkinson, Stroke)를 각각 받는다.

### 4.2 보관 원칙 [필수]
```
data/raw/        # 공식 원본. 읽기 전용. 수정·덮어쓰기 금지. Git 제외
data/interim/    # 파싱·QC 결과
data/processed/  # Session 단위 feature + label 테이블
```

### 4.3 Data manifest [필수]
Session마다 다음을 기록한다. 구현: `src/kimore/qc.py` → `data/interim/manifest_sessions.csv`, `manifest_files.csv`

| 필드 | 설명 |
|---|---|
| subject_id, group, exercise | 식별 정보 (feature 아님) |
| has_position / has_orientation / has_timestamp | 파일 존재 여부 |
| has_clinical / has_suppinfo | 라벨 파일 존재 여부 |
| n_rows_raw, n_frames | 원본 행 수, 유효 프레임 수 |
| file_hash | 파일별 SHA-256 (`manifest_files.csv`) |
| duplicate_of | 내용이 같은 복사본 식별 |
| misplaced | 파일명의 Subject ID와 폴더의 Subject ID가 다른 경우 |
| qc_status | `ok` / `warning` / `excluded` + 사유 |

- 인원을 셀 때는 **복사본을 제외하고, 파일명이 아닌 정규화된 Subject ID 기준**으로 센다.

### 4.4 Raw 신호 QC [필수]
- Position과 Timestamp의 행 수 일치
- Timestamp: 단위 확인, frame interval 분포, 중복 timestamp, frame drop
- Tracking state
  - 관측값: 2와 1 [D]. 값의 의미(2 = tracked, 1 = inferred)는 Kinect SDK 기준 해석 [E] → 원본 문서로 확인
  - 확인한 Position 파일 40개 기준 상태값 1의 비율: 약 6.3%, 0의 비율: 0% [D]
  - Session별·joint별 비율을 QC 지표로 기록한다. **ML feature로는 쓰지 않는다.**
- 이상치: 프레임 간 급격한 위치 변화, 뼈 길이의 비정상 변동
- Orientation: 확인한 파일 기준으로 joint 3, 15, 19, 21, 22, 23, 24는 모든 프레임에서 4값이 0이다 [D]. Script 주석 `no orientation`과 일치한다 [S].
  → **결측이 아닌 구조적 값**으로 취급하고 feature 계산에서 제외한다.

### 4.5 SuppInfo QC [필수]
**현재 확인 결과** (복사본 제외, Subject ID 기준) [D]

| Group | 대상자 폴더 | SuppInfo 있음 | Age 있음 |
|---|---|---|---|
| Expert | 17 | 17 | 17 |
| NotExpert | 27 | **23** | 23 |
| BackPain | 8 | 8 | 8 |
| Parkinson | 16 | **12** | 12 |
| Stroke | 10 | **9** | 9 |
| **합계** | **78** | **69** | **69** |

- SuppInfo가 없는 대상자의 값을 보정하거나 추정하지 않는다. [필수]
- Group은 폴더 구조로도 알 수 있다. 두 값이 일치하는지 확인한다.

### 4.6 Clinical Label QC [필수]

**검사 항목**
- 결측
- 범위: TS 0~50, PO 0~15, CF 0~35
- TS와 PO+CF의 관계
- 값 형식: 정수 / 1/3 단위 / 불규칙 소수
- 같은 대상자의 여러 폴더에 있는 라벨 파일끼리 내용 일치, 복사본, 다른 폴더에 잘못 놓인 파일

**"점수 보유 대상자" 집계 기준** — 이 문서에서는 항상 기준을 함께 표기한다.

| 기준 | Expert | NotExpert | BackPain | Parkinson | Stroke | 합계 |
|---|---|---|---|---|---|---|
| C1. ClinicalAssessment 파일이 있는 대상자 | 15 | 25 | 8 | 12 | 5 | **65** |
| C2. 점수가 1개 이상 있는 대상자 | 14 | 25 | 8 | 12 | 5 | **64** |
| C3. 15개 점수가 모두 있는 대상자 | 14 | 25 | 7 | 12 | 5 | **63** |

(복사본 제외, Subject ID 기준 [D])

**현재 확인된 사실** [D]
| 항목 | 내용 |
|---|---|
| 전체 결측 | E_ID17 (파일은 있으나 15개 점수 모두 비어 있음) |
| 부분 결측 | B_ID8 — Ex4의 TS·PO·CF |
| 범위 이탈 | 없음 |
| TS ≠ PO+CF | E_ID3, E_ID4의 Ex5 (차이 0.67). 그 외에는 확인한 범위에서 TS = PO + CF |
| 값 형식 (C2 기준 64명) | 정수형 32 · 1/3 단위 27 · 불규칙 소수 5 (B_ID3, B_ID4, P_ID10, S_ID5, S_ID6) |

- 1/3 단위 값이 평가자 3명의 평균인지는 **확인 필요** [E].
- 위 항목을 곧바로 오류로 단정하지 않는다. 원본을 재확보한 뒤 다시 확인한다.
- 출처가 불분명한 라벨은 필요하면 포함 분석과 제외 분석을 비교하는 **민감도 분석**을 한다.
- TS = PO + CF는 "대부분 성립"으로 기술한다.

### 4.7 Age 분포 [D]
**Age 분석 대상은 SuppInfo에 Age가 실제로 있는 대상자 69명이다.** 전체 대상자는 78명이다.

| Group | 전체 | Age 분석 대상 | 평균 나이 | 범위 |
|---|---|---|---|---|
| Expert | 17 | 17 | 33.5 | 22–57 |
| NotExpert | 27 | 23 | 35.9 | 24–76 |
| BackPain | 8 | 8 | 53.6 | 30–78 |
| Parkinson | 16 | 12 | 66.0 | 53–81 |
| Stroke | 10 | 9 | 59.3 | 17–81 |

### 4.8 점수 분포 (TS, C2 기준, Session 단위) [D]
| Group | TS 값 수 | TS ≥ 45 비율 |
|---|---|---|
| Expert | 70 | 46% |
| NotExpert | 125 | 47% |
| BackPain | 39 | 15% |
| Parkinson | 60 | 7% |
| Stroke | 25 | 0% |

→ 건강군에 천장효과 가능성이 있고, 전체 분산 중 그룹 간 차이의 비중이 클 수 있다.

---

## 5. 분석 단위 [필수]

- ML 기본 표본 = **Session (Subject × Exercise)**
- Frame과 Repetition은 feature를 만드는 내부 데이터이며 독립 표본이 아니다.
- train/validation/test 분할은 **Subject 단위**로 한다. 같은 Subject의 다른 Exercise나 Session이 서로 다른 split에 들어가면 안 된다.
- 기본 모델은 **운동별 모델**이다. Ex1~Ex5를 무조건 합치지 않는다.
  - 운동별로 feature dictionary, feature 계산, target, 분석, 모델을 정의한다.
  - 운동 통합 모델은 공통 feature를 정의할 수 있을 때만 탐색 분석으로 다룬다.
- 운동별 포함 대상자는 데이터 확보 후 결정한다.

---

## 6. Feature Engineering

### 6.1 원저자 Script에서 확인된 사실
| 내용 | 근거 |
|---|---|
| `main_Ex1.m`은 `preproc_Ex1` → `feat_extract_Ex1` 순서로 실행한다 | [S] Ex1만 확인 |
| Ex2~Ex5도 같은 이름 패턴의 파일이 있다 | [D] 실행 구조는 확인 필요 |
| `feat_extract_Ex1.m`에 `%% PRIMARY OUTCOME` 구간과 `% Filtering C.F.` 구간이 있다 | [S] |
| Ex1 PRIMARY OUTCOME 신호: `atan2(Spine_Sh(:,3)-Elbow(:,3), Spine_Sh(:,2)-Elbow(:,2))`를 도(°)로 변환, 좌·우 각각 계산 | [S] |
| 인접 프레임 차이가 ±100°를 넘으면 부호를 반전하는 특이점 처리 | [S] |
| C.F. 구간에서 필터링하는 변수 (Ex1): `angologomito*`, `angologinocchio*`, `angleHip*`, `sommaaree*`, `link_hand*`, `link_foot*` | [S] 변수명만 확인 |
| Ex5 필터링 변수: `assezshoulder*`, `assexshoulder*`, `link_shoulderhand*`, `link_foot*`, `sommaaree*`, `link_hip*`, `link_shoulder*`, `link_hand*`, `deltayknee*` | [S] 변수명만 확인 |

### 6.2 추정이므로 확인이 필요한 내용 [E]
- Ex1 PRIMARY OUTCOME 각도의 임상적 의미(어떤 관절 운동인지): **임상적 의미 확인 필요**
- C.F. 변수명의 의미(예: `angologomito`가 팔꿈치 각도인지): **원저자 Script에서 계산식을 확인한 뒤 명시**
- 각 운동의 실제 동작과 P.O./C.F. 신호의 대응 관계: **확인 필요**
- → 위 내용은 Gate 4(원저자 Script 해독)에서 계산식 기준으로 확정한다.

### 6.3 설계 원칙 [필수]
- 원저자의 P.O./C.F. 구분을 출발점으로 삼는다. 각 신호의 의미는 계산식을 확인한 뒤 feature dictionary에 기록한다.
- **모델 하나당 feature 약 6~10개를 기본 상한**으로 둔다.
- feature를 많이 만든 뒤 고르는 방식으로 진행하지 않는다.
- 운동 특화 feature는 해당 운동에서만 쓴다.
- 해석 근거가 불분명한 feature(예: 평균 각도)는 근거가 있을 때만 쓴다.

### 6.4 후보 유형 (운동별 적용 여부는 Gate 6에서 결정)
| 후보 | 필요 데이터 | 주의점 |
|---|---|---|
| ROM (반복별 max − min) | Position, 반복 분할 | 준비 자세 구간 제거 |
| 주요 각도 (peak 등) | Position | 특이점 처리 |
| 각속도 / 최대 각속도 | Position, Timestamp | Timestamp 단위 확인 전에는 계산하지 않음. 필터 영향 (§7.3) |
| 반복 시간, 반복 간 변동성 | 반복 분할, Timestamp | 반복 수가 적으면 불안정 |
| 좌우 대칭성 | Position | 운동별 정의 필요 |
| 보상 움직임 | Position | 원저자 C.F. 신호 확인 후 정의 |
| 운동 특화 feature (예: 스쿼트 깊이) | Position | 해당 운동이 무엇인지 확인한 뒤에만 |
| 부드러움 (SPARC, jerk) | Position, Timestamp | [선택]. 필수 아님 |

### 6.5 Feature dictionary [필수]
운동별로 feature마다 다음을 기록한다.
`name / exercise / PO·CF 대응 / 계산식 / 입력 joint / 필터 설정 / 단위 / 의미 / 근거([S]/[D]/[E]) / 주의점 / 원저자 대응 변수`

---

## 7. 전처리

### 7.1 파싱 [필수]
- 헤더 없음 → §3.3 schema로 열 이름 부여
- 앞쪽 빈 행 제거 (행 수를 고정하지 않고 빈 행을 탐지해 제거)
- 행 끝 쉼표로 생기는 빈 열 제거
- Position·Orientation·Timestamp 프레임 정렬

### 7.2 결측·추적 상태·이상치
- 처리 규칙은 원본 QC 결과를 본 뒤 결정한다.
- 구조적 0값(orientation)은 결측 처리하지 않는다.

### 7.3 Filtering
- 원저자 `filtering.m`의 실제 코드 [S] (확인한 사본 1개 기준, 사본 간 동일성은 확인 필요):
  - `butter(3, (2*1)/30, 'low')` + `filtfilt` → Butterworth 저역통과, 3차, cutoff 1 Hz, fs = 30, zero-phase
  - 파일 주석은 "이동평균"이라고 되어 있으나 코드 기준으로 기록한다.
- fs = 30은 Script의 설정값이다 [S]. 실제 프레임 간격은 Timestamp로 별도 확인한다.
- 위치·각도 feature: 원저자 필터를 출발점으로 한다.
- 미분·속도·부드러움 feature: 별도 필터 설정을 허용한다. 이 경우 종류, cutoff, sampling rate, 선택 근거를 문서화한다.
- **최종 필터 설정은 데이터 확보 후 결정한다.**

### 7.4 좌표·신체 크기 정규화
신체 기준 좌표계와 신체 크기 정규화를 검토한다. 방법은 Gate 5에서 결정한다.

### 7.5 Orientation
- 필수가 아니다. 구조적 0값 관절을 제외한 나머지의 품질을 원본 확보 후 평가한다.
- 품질이 부족하면 Position + Timestamp 기반 분석만 한다.
- **사용 여부는 데이터 확보 후 결정한다.**

### 7.6 반복 분할
- 원저자 방식 [S] (Ex1에서 확인):
  - `MINPEAKHEIGHT = max(signal) / sqrt(2)`
  - `MINPEAKDISTANCE = floor(nsamples / 12)`
- Ex2~Ex5의 방식은 확인 필요.
- 원본 확보 후 Python으로 재현하고 Session별로 시각적 검증을 한다. [필수]
- **최종 파라미터는 데이터 확보 후 결정한다.**

---

## 8. Leakage Prevention [필수]

### 8.1 방지 대상
| 유형 | 방지 방법 |
|---|---|
| Subject leakage | Subject 단위 분할 |
| Frame / repetition leakage | Session 단위로 집계한 뒤에만 표본 생성 |
| Imputation / scaling / feature selection / PCA leakage | `Pipeline` 안에서 fold별로 fit |
| Target leakage | TS가 target이면 PO·CF를 feature로 쓰지 않음. 다른 운동의 점수도 feature로 쓰지 않음 |
| Metadata / Group leakage | Subject ID, Group, 폴더명, 파일명은 feature 금지 |
| 대리 변수 | 녹화 날짜, 프레임 수, 세션 길이, 카메라 거리, tracking state 비율은 feature 금지 (QC 전용) |
| Age confounding 무시 | §10.3 상시 수행 |
| 분석 선택 leakage | 주 분석 사전 지정 (§11.2) |

### 8.2 CV 안과 밖의 구분
- **CV 밖 허용** (Session 내부의 결정론적 처리): parsing, filtering, 좌표 변환, 반복 분할, Session 단위 feature 계산
- **CV 안에서 fit 필수** (데이터에서 학습되는 단계): 통계량 기반 imputation, scaling, feature selection, PCA, 모델 학습, 하이퍼파라미터 튜닝

---

## 9. Cross Validation

### 9.1 확정 사항 [필수]
1. train/test는 Subject 단위로 분리한다.
2. 같은 Subject가 여러 fold에 들어가지 않는다.
3. 확정된 fold assignment(`subject_id → fold`)는 파일로 저장하고 모든 실험이 같은 파일을 쓴다.
4. 튜닝을 하면 Nested CV를 쓴다.

### 9.2 데이터 확보 후 결정
- CV 방식 (후보: StratifiedGroupKFold, GroupKFold, Leave-One-Subject-Out)과 fold 수
- 근거: 현재 Stroke는 C1 기준 5명이다.

---

## 10. 통계 분석

### 10.1 수행 항목 [필수]
기술통계 · 점수 분포·천장효과 · Spearman correlation · Mann-Whitney U · Kruskal-Wallis · 효과크기 · 다중비교 보정

### 10.2 보고 원칙 [필수]
p-value만 강조하지 않는다. 효과크기, 신뢰구간, 표본 수(어떤 집계 기준인지 포함)를 함께 보고한다.

### 10.3 Age confounding [필수]
Age를 feature로 쓰는지와 관계없이 수행한다. 분석 대상은 Age가 실제로 있는 대상자로 한정하고 그 수를 함께 보고한다.
- Group별 Age 분포
- Age와 target의 관계
- Age-only baseline
- Age 보정 분석
- 가능하면 연령 매칭 또는 연령 제한 부분집합 분석
- Group 차이를 곧바로 운동 수행 능력 차이로 해석하지 않는다.

### 10.4 Reliability [중요]
반복 측정이 충분하면 반복 간 ICC 또는 CV를 검토한다. 가능 여부는 데이터 확보와 반복 분할 검증 후 판단한다.

---

## 11. ML Modeling

### 11.1 문제 정의
- 주 분석: 회귀 (PO/CF 점수 추정), 운동별 모델 [필수]
- 보조 분석 [선택]: 건강군 vs 환자군 이진 분류. Age-only baseline과 비교 필수
- 5개 Group 분류는 범위에서 제외

### 11.2 주 분석과 탐색 분석 [필수]
- Gate 2 이후 **주 분석 1개를 사전 지정**한다 (target, 운동, feature set, 모델, 지표).
- 나머지 조합은 탐색 분석으로 표기한다.
- 사전 지정 내용은 분석 전에 docs/에 기록하고, 바꾸면 사유를 남긴다.
- **최종 target과 주 분석은 데이터 확보 후 결정한다.**

### 11.3 Baseline [필수]
| # | Baseline | 목적 |
|---|---|---|
| B1 | 전체 평균 | 최소 기준 |
| B2 | Group 평균 (참조용) | Group만 알 때보다 움직임 feature가 추가 정보를 주는지 판단. 후보 모델에 Group을 넣는 것이 아님 |
| B3 | Age-only | 연령 교란 수준 확인 (Age가 있는 대상자 한정) |
| B4 | feature 1~2개 선형 모델 | 단순 지표 대비 이득 확인 |

### 11.4 후보 모델
- [필수] Ridge, 트리 기반 모델 1개
- [선택] ElasticNet, 추가 트리 모델
- 하이퍼파라미터 탐색 범위는 작게 유지한다.

### 11.5 최종 모델 선정 기준 [필수]
- outer CV의 MAE가 실질적으로 비슷하면 더 단순하고 해석 가능한 모델을 고른다.
- "실질적으로 비슷함"의 수치 기준은 주 분석을 지정할 때 함께 정한다.
- **최종 모델은 ML 검증 후 결정한다.**

### 11.6 모델 저장
- 선택된 설정으로 전체 데이터에 다시 학습시켜 저장하고, 성능은 outer CV 결과로 보고한다.
- model configuration, feature 목록, manifest 해시, 버전을 함께 기록한다.

---

## 12. Evaluation

### 12.1 회귀 [필수]
- 주 지표: **MAE** (점수 단위)
- 보조: Spearman correlation, RMSE, R² (R²는 단독 근거로 쓰지 않음)
- 전체 성능과 Group 내부 성능을 구분해 보고한다.
- 모든 지표는 baseline B1~B4와 나란히 제시한다.

### 12.2 보조 분류 (수행할 경우)
balanced accuracy, macro-F1, ROC-AUC, confusion matrix. Accuracy 단독 주장 금지.

### 12.3 시각화
실제값 vs 예측값, Bland-Altman plot

### 12.4 민감도 분석
출처가 불분명한 라벨을 포함한 경우와 제외한 경우. 필터 설정이 여러 개면 설정 간 비교.

---

## 13. Reproducibility [필수]
고정하고 기록할 항목: random seed · Python version · package versions · requirements 파일 · data manifest · file hash · fold assignment · feature dictionary · model configuration · 주 분석 사전 지정 문서

### 13.1 실행 환경
- Python 3.12.10 (winget 설치, 사용자 범위) + 프로젝트 `.venv`
- 패키지 버전: `requirements.txt` (현재 numpy, pandas, openpyxl, pytest)
- MATLAB / Octave: 미설치. 설치가 필요하면 사용자에게 먼저 보고하고 승인을 받는다.

### 13.2 원저자 Script 검증 방식 (데이터 확보 후 결정)
1. GNU Octave를 설치해 원저자 Script를 실행하고 Python 결과와 비교
2. 원저자 수식과 파라미터를 Python으로 재현하고 일부 샘플을 수작업으로 검증

---

## 14. 테스트 (tests/) [필수]

### 14.1 설계 원칙
- 테스트는 **데이터 구조와 분석 규칙이 깨졌는지**를 검증한다.
- 특정 분석 결과 수치(예: Session 수, 대상자 수)를 하드코딩하지 않는다.
- 데이터가 바뀌어도 유효한 불변 조건(invariant)만 검사한다.
- 잘못된 형식은 테스트용 소형 합성 fixture로 검증한다. 원본 데이터를 테스트 fixture로 저장소에 넣지 않는다.

### 14.2 데이터 파싱 테스트
- 파싱 결과가 25 joints schema(이름, 순서, joint당 x/y/z/state)와 일치하는지
- Position 유효 열 수 = 25 × 4
- Orientation 유효 열 수 = 25 × 4
- Timestamp 행 수 = Position 프레임 수 (불일치하면 탐지되는지)
- 잘못된 형식을 탐지하는지: 열 수 이상, 숫자가 아닌 값, 빈 파일, 행 끝 쉼표 유무 차이, 빈 행 위치 변화

### 14.3 Label QC 테스트
- TS 0~50, PO 0~15, CF 0~35 범위를 벗어난 값을 탐지하는지
- 결측을 탐지하는지
- TS와 PO+CF의 불일치를 허용 오차 기준으로 탐지하는지
- 복사본과 잘못 놓인 파일을 탐지하는지

### 14.4 Leakage / CV 테스트
- 같은 Subject가 여러 fold에 들어가지 않는지
- 각 split에서 train과 test의 Subject가 겹치지 않는지
- fold assignment에 중복이나 누락된 Subject가 없는지
- Pipeline의 feature 목록에 금지 컬럼(Subject ID, Group, 대리 변수, target 계열 점수)이 없는지

(규칙 검증 함수는 `src/kimore/cv.py`에 구현됨. 실제 splitter와 fold 생성은 Gate 3에서 결정·구현한다.)

---

## 15. 데이터 이용 조건 및 인용 [필수]

- KIMORE 공식 이용 조건을 확인한다. → **공식 배포처의 이용 조건 확인 필요**
- 원본 데이터 재배포 금지 여부를 확인한다. → **공식 배포처의 이용 조건 확인 필요**
- 논문과 데이터셋의 공식 인용 형식을 확인한다. → 논문 서지 정보는 README에 기록함. 데이터셋 인용 형식은 **공식 배포처의 이용 조건 확인 필요**
- README에 데이터 출처와 인용 정보를 기록한다.
- 영상 등 개인정보가 포함될 수 있는 원본 자료는 Git 저장소에 포함하지 않는다.
- `data/raw/` 전체는 Git에서 제외한다 (`.gitignore`).
- 이용 조건을 확인하기 전에는 데이터를 외부에 업로드하거나 재배포하지 않는다.
- 이용 조건 확인 결과는 확인 일자와 출처와 함께 docs/에 기록한다.

---

## 16. 서비스화 범위 [후순위]

Gate 8을 통과하고 ML 결과가 충분할 때만 검토한다.

| 구성 | 도입 조건 |
|---|---|
| FastAPI `POST /predict` | 최종 pipeline 확정 후. 입력 형식(25-joint 시계열 또는 검증된 feature vector)은 그때 결정. 출력은 예측 PO/CF, model version, 주요 feature 정보 |
| DB | 예측 이력, 모델 버전 관리, 세션 결과 관리, 사용자별 결과 저장 같은 실제 기능 요구가 있을 때만. CSV를 DB로 옮기는 목적은 금지 |
| Spring Boot / React | 서비스 기능이 정의된 뒤 선택적으로 |

### 16.1 프로젝트 구조
```
data/{raw, interim, processed}/
notebooks/
src/kimore/   io.py, labels.py, qc.py, cv.py(규칙 검증)  (preprocess, segment, features는 해당 Gate에서 추가)
models/
tests/
docs/
```
프로젝트가 커질 때만 모듈을 폴더로 분리한다.

---

## 17. Gate / 개발 순서

| Gate | 작업 | 통과 조건 |
|---|---|---|
| 0 | 원본 재확보 + 실행 환경 준비 + 이용 조건 확인 | 완전한 원본 확인, 분석 가능 Session 수 확인, Python 환경 준비(완료), Octave 필요 여부 결정, 이용 조건 기록 |
| 1 | Inventory + parsing + raw QC + 파싱 테스트 | manifest 완성, 분석 가능 Session 목록 확정, schema 재검증, 파싱 테스트 통과 |
| 2 | Label·SuppInfo QC + EDA + Label QC 테스트 | 집계 기준별 인원 확정, target 후보 확정, 주 분석 1개 사전 지정, Age confounding 분석 계획 확정 |
| 3 | Subject 단위 fold 설계 + CV 테스트 | CV 방식 확정, fold assignment 저장, CV 테스트 통과 |
| 4 | 원저자 Script 해독·재현 | 계산식 기준으로 신호 의미 확정 ([E] → [S]), 핵심 처리 로직 검증 |
| 5 | 전처리 + 반복 분할 | 반복 분할 시각적 검증, 실패 Session 기록 |
| 6 | Feature engineering | 운동별 feature dictionary 완성, 상한 준수 |
| 7 | 통계 분석 | 효과크기 포함 결과, Age confounding 검토 |
| 8 | Baseline → Ridge → Tree | baseline 대비 개선 여부, Nested CV 결과, 사전 기준 적용 |
| 9 | 최종 평가 | 전체·Group 내부 성능, 민감도 분석, 시각화 |
| 10 | 모델 카드 + README | 한계, Kinect v2 의존성, 표본 수 한계, 진단 목적이 아님, 데이터 출처·인용 명시 |
| 11+ | (조건부) FastAPI → 필요하면 DB → Spring Boot / React | Gate 8 결과가 충분할 때만 |

현재 로컬 사본으로 진행하는 Gate 1·2·4·5의 결과는 **개발용·예비**로 표시하고, 원본 재확보 후 같은 코드로 다시 실행한다.

---

## 18. 명시적 제외 범위
- 질병 진단·판별 표현과 기능
- 웹캠이나 다른 포즈 추정기에서도 같은 성능이 나온다는 주장
- 작은 표본으로 임상적 효능을 확정하는 주장
- Accuracy 단독 성능 주장
- Subject leakage가 생기는 랜덤 분할
- 불완전한 현재 로컬 데이터로 최종 모델 평가
- 5개 Group 분류를 핵심 분석으로 삼는 것
- 목적 없는 DB 도입, 의미 없는 기술 스택·알고리즘 추가
- 결측 인구통계·라벨 값의 임의 보정 또는 추정
- 변수명만 보고 임상적 의미를 확정하는 것
- 영상 파일 사용 및 저장소 포함, 이용 조건 확인 전 데이터 외부 업로드·재배포
- 원본 데이터 수정·덮어쓰기

---

## 19. 프로젝트 한계
- 표본 수: 전체 78명이고 라벨 보유는 C1 기준 65명이다. 일부 그룹은 매우 작다(BackPain 8명, Stroke C1 기준 5명).
- 연령 교란: Age 분석 대상 69명 기준으로 건강군 평균 약 34~36세, 환자군 평균 약 54~66세다.
- 천장효과: 건강군 세션의 약 절반이 TS ≥ 45다.
- 라벨 출처: 값 형식이 섞여 있고 출처가 불분명한 값이 있다.
- 센서 의존성: Kinect v2 25-joint 데이터에 한정된 결과다.
- 측정 정밀도: 관절 추정 오차와 가려짐이 있다.
- 단일 데이터셋: 외부 검증 데이터가 없다.
- 통계적 연관이나 예측 성능은 임상적 효능이나 진단 능력을 뜻하지 않는다.
