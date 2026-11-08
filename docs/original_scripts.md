# 원저자 MATLAB Script 해독 (Gate 4, 예비)

> 작성일: 2026-11-08 · 대상: 현재 로컬 사본 `data/raw/**/Script/*.m` · 읽기 전용 분석
> 근거 표기: **[S]** 코드(또는 코드 안의 저자 주석)에서 직접 확인 · **[E]** 변수명·주석 번역·계산식으로부터의 해석·추정
> 임상적 의미는 모두 [E]이며 **임상적 의미 확인 필요**. 원본 재확보 후 이 문서를 다시 검증한다.

---

## 1. 사본 간 동일성 (MD5, `(1)` 사본 포함)

| 파일 종류 | 사본 수 | 서로 다른 버전 수 | 대표 사본 |
|---|---|---|---|
| filtering.m | 149 | 1 | `CG/Expert/E_ID1/Es1/Script/filtering.m` |
| main_Ex1~5.m | 41 / 26 / 36 / 23 / 24 | 각 1 | `CG/Expert/E_ID11/Es1~3`, `E_ID10/Es4`, `E_ID12/Es5` |
| preproc_Ex1~5.m | 36 / 30 / 24 / 32 / 37 | 각 1 | `E_ID13/Es1`, `E_ID12/Es2`, `E_ID14/Es3`, `E_ID10/Es4`, `E_ID13/Es5` |
| feat_extract_Ex1~5.m | 30 / 36 / 31 / 26 / 22 | 각 1 | `E_ID1/Es1`, `E_ID10/Es2`, `E_ID1/Es3`, `E_ID14/Es4/feat_extract_Ex4(1).m`, `E_ID11/Es5` |

- 같은 이름의 파일은 `(1)` 사본까지 포함해 **모두 바이트 단위로 동일**하다 [S].
- `preproc_Ex1~5.m`은 해시는 서로 다르지만(Ex2·Ex4·Ex5는 동일 해시), **공백과 파일 끝 줄바꿈을 제외하면 내용이 같다** [S]. 즉 5개 운동이 같은 전처리를 쓴다.
- 각 스크립트는 자기 운동 폴더(EsN)에만 있다(폴더-운동 불일치 없음) [S].
- `Script/Script/` 중첩 폴더가 45개 있다(예: `E_ID10/Es2/Script/Script/feat_extract_Ex2.m`). 내용은 위 대표본과 같은 해시 그룹에 속한다 [S]. 압축 해제·복사 과정의 부산물로 보인다 [E].

---

## 2. 공통 구조

### 2.1 실행 순서 [S]
`main_ExN.m`:
```matlab
clc;clear;close all;
addpath('../Raw')
%Joint_Position=csvread('JointPosition...');      % 주석 처리됨
%Joint_Orientation=csvread('JointOrientation...'); % 주석 처리됨
preproc_ExN
feat_extract_ExN
```
- 데이터 로딩 줄이 **주석 처리**되어 있어, `Joint_Position` 변수를 사용자가 미리 만들어 두어야 실행된다 [S].
- `preproc`, `feat_extract`는 함수가 아닌 **스크립트**(작업공간 공유)다 [S].
- 결과를 파일로 저장하는 코드(`save`, `xlswrite`, `csvwrite`, `writetable`)가 없다 [S]. 결과는 MATLAB 작업공간의 변수(시계열과 피크 값·위치)로만 남는다.
- **Joint_Orientation과 TimeStamp는 어떤 feat_extract에서도 쓰이지 않는다** [S]. 모든 신호가 Position에서 계산되고, 시간은 샘플 번호로만 다룬다.
- 점수(TS/PO/CF)를 계산하는 코드는 없다. Ex5에 `Temporalscore5(..., 71, 92, 50, ...)` 호출이 **주석으로만** 남아 있다 [S]. 이 함수 파일은 데이터에 없다 [S].

### 2.2 preproc (공통) [S]
- `index_<Joint>` = 1, 5, …, 97 (joint당 4열, MATLAB 1-based). 주석 `% no orientation`: Head, Foot_Left, Foot_Right, Tip_Left, Thumb_Left, Tip_Right, Thumb_Right.
- 각 관절을 `[col, col+1, col+2]` 3열로 잘라 변수로 만든다: `Hip_R/L, Knee_R/L, Ankle_R/L, Foot_R/L, Spine_B, Spine_M, Head_C, Spine_Sh, Shoulder_R/L, Elbow_R/L, Wrist_R/L, Hand_R/L`.
- 4번째 열(tracking state)은 쓰지 않는다 [S].

### 2.3 좌표축
- 첫 줄 주석 "in tabella di dati ho le xyz di ogni giunto"(데이터 표에 각 관절의 xyz가 있다) → 열 +0/+1/+2 = x/y/z [S, 저자 주석].
- 각 축이 Kinect 카메라 좌표의 무엇(x 좌우, y 상하, z 깊이)인지는 코드에 없다 → [E]. 아래 "평면" 해석도 모두 이 [E]에 기댄다.
- 코드 안에서는 `(:,1)=x`, `(:,2)=y`, `(:,3)=z`로 일관되게 쓰고, 일부 C.F.는 `index_X+1`(y), `index_X+2`(z)를 직접 읽는다 [S].

### 2.4 필터 (`filtering.m`) [S]
```matlab
filtCutOff = 1; sample = 30;
[b, a] = butter(3, (2*filtCutOff)/sample, 'low');
media = filtfilt(b, a, x);
```
- 3차 Butterworth 저역통과, 정규화 차단 2·1/30, zero-phase(`filtfilt`). 주석은 "이동평균"이지만 코드는 Butterworth다.
- fs = 30은 코드 상수다. 실제 프레임 간격(Timestamp)과는 연결되어 있지 않다.

### 2.5 공통 C.F. 계산식 (여러 운동에서 반복) [S]
| 변수 | 계산식 | 사용 운동 |
|---|---|---|
| `angologomitoL/R` | 코사인 법칙: 변 \|Shoulder–Elbow\|, \|Elbow–Wrist\|, \|Shoulder–Wrist\| → Elbow 꼭짓점의 각(°) | Ex1–4 |
| `angologinocchioL/R` (C.F.) | 코사인 법칙: \|Hip–Knee\|, \|Knee–Ankle\|, \|Hip–Ankle\| → Knee 꼭짓점의 각(°) | Ex1–4 |
| `angleHipL / angoloHipL` | `atan2(SpineBase.x − HipL.x, SpineBase.y − HipL.y)` (°) | Ex1–3 |
| `angleHipR / angoloHipR` | `atan2(HipR.x − SpineBase.x, HipR.y − SpineBase.y)` (°) | Ex1–3 |
| `angoloascellaL` | 코사인 법칙, 삼각형 (ShoulderL, ElbowL, **HipR**) 의 ShoulderL 꼭짓점 각 | Ex2–4 |
| `angoloascellaR` | 같은 방식, 삼각형 (ShoulderR, ElbowR, **HipL**) | Ex2–4 |
| `sommaaree` | 헤론 공식 삼각형 2개 합: (HipL, HipR, ShoulderR) + (ShoulderL, ShoulderR, HipL) | Ex1, Ex5 |
| `link_hand` | \|HandR − HandL\| (3D) | Ex1, Ex2, Ex5 |
| `link_foot` | \|AnkleR − AnkleL\| (3D) | Ex1, Ex5 |
| `link_shoulder` | \|ShoulderR − ShoulderL\| (3D) | Ex2, Ex4, Ex5 |
| `link_hip` | \|HipR − HipL\| (3D) | Ex2, Ex3, Ex5 |
| `der*` ("REGOLARITA'") | `diff(최대 피크 인덱스)` = 연속 최대값 사이 샘플 수 | 전 운동 |

- C.F. 신호 대부분은 **끝 15샘플을 잘라낸 뒤**(`(1:end-15)`) 필터링한다. 주석: "acquisizione file 관련 마지막 15개 값 제거" [S].
- 의미 해석 [E] — **임상적 의미 확인 필요**: `gomito`=팔꿈치, `ginocchio`=무릎, `ascella`=겨드랑이, `aree tronco`=몸통 면적(주석 "calcolo area tronco metodo erone"), `ANGOLO C`의 주석은 "엉덩이와 아래 척추 사이 각(C각의 절반)". `angoloascella`의 반대쪽 Hip 사용이 의도인지는 확인 필요.

---

## 3. 운동별 정리

운동 이름은 Script 어디에도 없다. 아래 "추정 동작"은 모두 [E]이며 **공식 문서로 확인 필요**하다.

### Ex1 (`feat_extract_Ex1.m`, 145행)
**P.O.** [S]
- `shoulder_extR = atan2(Spine_Sh.z − Elbow_R.z, Spine_Sh.y − Elbow_R.y)` × 180/π, 좌측도 동일(Elbow_L).
- 특이점 처리: 인접 샘플 차이가 −100 미만 또는 100 초과면 다음 값의 부호를 뒤집는다.
- 필터: `filtering` (앞·뒤 자르기 없음).
- 피크: 최대값 `findpeaks(sig, MINPEAKHEIGHT=max(sig)/√2, MINPEAKDISTANCE=floor(n/12))`. 최소값은 `max(sig) − sig`에 **같은 임계값 max(sig)/√2**와 같은 간격을 적용.
- [E] y–z 평면(시상면 추정)에서 SpineShoulder 대비 팔꿈치의 각도. 변수명 `shoulder_ext`, 주석 legend "underarm right angle". **임상적 의미 확인 필요**.

**C.F.** [S]: `angologomito`, `angologinocchio`, `angleHip`, `sommaaree`, `link_hand`, `link_foot` (모두 끝 15샘플 제거 후 필터), `derR/derL`.

### Ex2 (`feat_extract_Ex2.m`, 200행)
**P.O.** [S]
- `angolobustoL = atan2(Shoulder_L.x − Hip_L.x, Shoulder_L.y − Hip_L.y)` (°), 우측 동일.
- 특이점 처리 코드는 **주석 처리**되어 있다.
- 필터 후 좌측만 부호 반전(`-filtering(...)`, 주석 "rescale angolo"), 이어서 앞 29샘플 제거(`30:end`).
- 피크: 최대값 임계 = `mean(sig)`, 간격 `floor(n/10)`; 최소값 임계 = `mean(max(sig)−sig)`, 간격 `floor(n/8)`.
- 활성 상태의 `figure; plot(angolobustoL)`이 남아 있다(일괄 실행 시 창이 뜸).
- [E] x–y 평면(전두면 추정)에서 Hip 대비 Shoulder의 기울기(`busto`=몸통). **임상적 의미 확인 필요**.

**C.F.** [S]: `angologomito`, `angologinocchio`, `angoloHip`, `angoloascella`(끝 15 제거, **NaN 샘플을 삭제** → 길이가 다른 신호와 달라짐), `assezR/L = Wrist.z − Spine_Shoulder.z` (앞 14·끝 15 제거, `15:end-15`), `assezhipR/L = Hip.z`, `assexhipR/L = Hip.x`, `link_hand`, `link_shoulder`, `link_hip`, `derR`.

### Ex3 (`feat_extract_Ex3.m`, 148행)
**P.O.** [S]
- `linkelbow = |Elbow_L.x − Elbow_R.x|` (각도가 아닌 x축 거리).
- 필터 후 `30:end-15`로 앞 29·끝 15샘플 제거.
- 피크: 최대값 임계 `mean(sig)`, 최소값 임계 `mean(max(sig)−sig)`, 간격 모두 `floor(n/20)`.
- [E] 양 팔꿈치의 좌우(x) 간격 변화 → 몸통 회전 동작의 대리 지표로 추정. **임상적 의미 확인 필요**.

**C.F.** [S]: `angolobracciospallaR/L` = 코사인 법칙, 삼각형 (Shoulder, Elbow, Spine_Sh)의 Shoulder 꼭짓점 각(자르기 없이 필터), `angologomito`, `angologinocchio`, `angoloHip`, `angoloascella`, `assezhipR/L = Hip.z`, `controlyR/L = 2·Wrist.y − Elbow.y − Shoulder.y` (`30:end-15`), `link_hip`, `der1`.

### Ex4 (`feat_extract_Ex4.m`, 169행)
**P.O.** [S]
- `variazioneZ = Spine_B.z − mean(Spine_B.z)`, `variazioneX = Spine_B.x − mean(Spine_B.x)` (평균 중심화된 SpineBase 위치).
- 필터 후 각각 `30:end-15`.
- 피크(Z, X 각각): 최대값 임계 `mean(sig)`, 최소값 임계 `mean(max(sig)−sig)`, 간격 `floor(n/20)`.
- [E] SpineBase(골반 추정)의 깊이(z)·좌우(x) 이동 → 골반 회전 동작 추정. **임상적 의미 확인 필요**.

**C.F.** [S]: `angologomito`, `angologinocchio`, `assezShoulderR/L = Shoulder.z`, `angoloascella`, `link_shoulder`, `deltazhip = |HipL.z − HipR.z|`, `deltayhip = |HipL.y − HipR.y|`, `deltazShoulder = |ShoulderL.z − ShoulderR.z|` (모두 끝 15 제거 후 필터), `derZ`.

### Ex5 (`feat_extract_Ex5.m`, 184행)
**P.O.** [S]
- `angologinocchioL = atan2(Hip_L.y − Knee_L.y, Hip_L.z − Knee_L.z) + atan2(Knee_L.y − Ankle_L.y, Ankle_L.z − Knee_L.z)` (°), 우측 동일. (C.F.용 코사인 법칙 무릎각과 **다른 식**)
- 앞 9샘플 제거(`10:end`) → 특이점 처리(±100) → 필터.
- 피크: 최대값 임계 `max(sig)/√2`, 최소값 임계 `max(max(sig)−sig)/√2`, 간격 `floor(n/10)`.
- 디버그용 출력 줄(`a=...`, `b=...`, 세미콜론 없음)이 남아 있다.
- [E] y–z 평면(시상면 추정)에서 대퇴·하퇴 분절 각의 합 → 무릎 굴곡(스쿼트 추정). **임상적 의미 확인 필요**.

**C.F.** [S]: `assezshoulderR/L = Shoulder.z`, `assexshoulderR/L = Shoulder.x`, `link_shoulderhand = |Shoulder − Hand|` (주석은 "spalla anca"=어깨-엉덩이로 잘못 적힘), `link_foot`, `sommaaree`, `link_hip`, `link_shoulder`, `link_hand`, `deltayknee = KneeR.y − KneeL.y` (모두 끝 15 제거 후 필터), `derR`, `derL`.

---

## 4. 최종 출력물 [S]
- 스칼라 feature 표는 만들지 않는다. 남는 것은 (1) 필터된 **시계열**, (2) 피크 값 배열(`maxR`, `minR` 등)과 위치(`ind_*`, `locs*`), (3) 피크 간격 배열(`der*`)이다.
- 이들을 임상점수로 바꾸는 단계(예: `Temporalscore5`)는 제공되지 않았다.

---

## 5. Python 재현 시 필요한 결정사항
1. **입력 로딩**: Script가 로딩을 하지 않으므로, 우리 파서(`kimore.io`) 결과를 `Joint_Position`과 같은 열 순서로 넘기는 규칙을 정한다.
2. **필터**: `scipy.signal.butter(3, 2*1/30, 'low')` + `filtfilt`로 재현할지. MATLAB `filtfilt`와 SciPy `filtfilt`의 가장자리 패딩 방식 차이(padlen 등)를 맞출지.
3. **fs**: 코드 상수 30을 그대로 쓸지, Timestamp에서 실제 간격을 계산해 쓸지.
4. **특이점 처리**: ±100° 부호 반전 규칙을 그대로 쓸지, `np.unwrap` 같은 다른 방식으로 바꿀지(Ex1, Ex5).
5. **앞·뒤 자르기**: 운동·신호마다 다른 자르기(앞 9/14/29, 끝 15)를 그대로 재현할지.
6. **피크 검출 파라미터**: MATLAB `findpeaks(MINPEAKHEIGHT, MINPEAKDISTANCE)` ↔ `scipy.signal.find_peaks(height, distance)`. 운동별 임계(`max/√2` vs `mean`)와 간격(n/8, n/10, n/12, n/20)을 그대로 쓸지. MATLAB은 거리 조건 적용 시 큰 피크를 우선하고 SciPy도 동일한 원칙이지만, 평탄 피크 처리 차이는 검증 필요.
7. **최소값 임계 불일치**: Ex1은 최소값에도 최대값 임계를 쓰고, Ex5는 반전 신호 기준 임계를 쓴다. 그대로 재현할지, 통일할지.
8. **Ex2 좌측 부호 반전**, **Ex2 `angoloascella` NaN 삭제(길이 불일치)** 를 그대로 둘지.
9. **acos 정의역**: 코사인 법칙 계산에서 수치오차로 |값|>1이 되면 NaN이 나온다. 클리핑할지.
10. **시계열 → 스칼라 feature 변환**: Script에 없으므로 우리가 정의해야 한다(feature dictionary).

## 6. 확인되지 않은 것
- 운동 Es1~Es5의 실제 동작 이름과 지시문
- x/y/z 축의 물리적 방향(카메라 좌표계)과 단위(미터 여부)
- 각 P.O./C.F. 신호의 임상적 의미, 그리고 PO/CF 점수 산정 규칙과의 대응
- `Temporalscore5` 등 점수화 함수의 내용과 임계값 71/92/50의 의미
- 끝 15샘플 제거의 이유("acquisizione file"의 구체적 의미)
- `angoloascella`에서 반대쪽 Hip을 쓴 것이 의도인지
- 원본 재확보 시 Script가 지금 사본과 같은 버전인지
