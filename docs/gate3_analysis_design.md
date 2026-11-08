# KIMORE Gate 3 진입 전 분석 설계

> 작성: 2026-11-09 · 상태: **설계 초안** (Gate 2 = PASS WITH CONDITIONS 이후)
> 근거 표기: **[S]** 원저자 Script 직접 확인 · **[D]** 데이터에서 직접 확인 · **[E]** 해석·추정 (확정 근거로 쓰지 않음)
> 이 문서의 수치는 현재 **로컬 사본** 기준이다. 표의 원천은 `python -m kimore.label_audit` 가 만드는 `reports/gate3/*.csv` 이며, 원본 재확보 후 같은 명령으로 다시 집계한다.
> 이 단계에서는 ML 학습, feature extraction, 분할 코드 구현을 하지 않는다.

---

## 1. Gate 2 결과 요약

상세: [label_eda_report.md](label_eda_report.md), [../reports/eda/EDA_report.md](../reports/eda/EDA_report.md)

| 항목 | 결과 [D] |
|---|---|
| Subject | 전체 78 · Age 있음 69 · Clinical 파일(C1) 65 · 점수 1개 이상(C2) 64 · Age+점수 56 |
| 점수 범위 이탈 | 0건 (TS 0–50, PO 0–15, CF 0–35) |
| TS ≠ PO+CF | 2 Session (E_ID3 Ex5, E_ID4 Ex5, 각 +0.67). 나머지는 TS = PO+CF |
| 상한 근처 (TS ≥ 45, Session) | 건강군 47% · 환자군 8%. 건강군 PO 만점 비율 34%, CF 17% |
| PO–CF | Spearman ρ 0.71 (Session n=319, 같은 Subject 관측 포함) |
| Age | 건강군·환자군 차이 큼 (rank-biserial r = −0.74, n=69). Age–TS ρ = −0.69 (Subject 평균, n=56), 건강군 안 −0.35, 환자군 안 −0.57 |
| Age overlap | 판정 B. 공통 구간 22–76세, 양쪽 모두 3명 이상인 10세 구간 2개, ±5세 1:1 매칭 환자 12/29명 |
| 라벨 민감도 | 불규칙 소수 5명 제외 시 작은 그룹 중앙값이 크게 움직임 (Stroke TS −6.2, BackPain TS +3.3). 전체 상관 구조 변화는 ρ < 0.1 |
| 신호 + 점수 Session | 운동별 3–7개 (아래 3절). **현재 사본으로는 모델 학습 불가** |

---

## 2. 원본 라벨 재검증 결과

방법: 대상 Subject 의 ClinicalAssessment 파일을 **모든 위치(다른 Exercise 폴더의 반복 저장본 포함)** 에서 openpyxl 로 셀 단위로 다시 읽었다 (원본 읽기 전용). 결과 파일: `reports/gate3/label_recheck_files.csv`, `label_recheck_values.csv`.

### 2.1 파일 단위 사실 [D]

| Subject | 파일 위치 (반복 저장) | 반복본 내용 | 셀 수식 | 셀 표시 형식 |
|---|---|---|---|---|
| B_ID3 | Es3 | 1개뿐 | 0 | `0` |
| B_ID4 | Es1 | 1개뿐 | 0 | `0` |
| P_ID10 | Es2, Es4, Es5 | 3개 모두 SHA-256 동일 | 0 | `0` |
| S_ID5 | Es3 | 1개뿐 | 0 | `0` |
| S_ID6 | Es1, Es3, Es5 | 3개 모두 SHA-256 동일 | 0 | `0` |
| E_ID3 | Es3, Es4, Es5 | 3개 모두 SHA-256 동일 | 0 | `0.00` |
| E_ID4 | Es3 | 1개뿐 | 0 | `0.00` |

- 모든 값은 **수식이 아닌 숫자 상수**로 저장되어 있다. 시트는 1개(`Foglio1`), 1행 헤더 + 1행 값 구조로 다른 Subject 와 같다.
- 셀 표시 형식 `0`(정수로 보이게 표시)은 불규칙 소수 5개 파일 모두에 있지만, **1/3 단위 값 파일 9개에도 있다**. 따라서 표시 형식으로 불규칙 소수 파일을 구분할 수 없다.
- 파일 메타데이터(작성자·수정일)는 불규칙 소수 파일과 다른 Clinical 파일이 같은 분포에 속한다 (모두 같은 작성자, 대부분 같은 수정일). 메타데이터로도 구분되지 않는다.
- 같은 Subject 의 반복 저장본끼리 내용 차이는 없다.

### 2.2 값 단위 사실 [D]

개별 Subject 의 점수 값은 이 문서에 옮기지 않는다 (데이터 이용 조건 확인 전, REQUIREMENTS §15). 값은 로컬 `reports/gate3/label_recheck_values.csv` 에만 둔다.

| 대상 | 원본 값의 형식 | 같은 Subject 다른 Exercise 와 비교 |
|---|---|---|
| B_ID3, B_ID4, P_ID10, S_ID5, S_ID6 | 15개 값 모두 불규칙 소수 (1/3 격자에 맞지 않는 긴 소수). 범위 이탈 없음 | 15개 모두 같은 형식. 5개 운동 모두 TS = PO+CF (차이 0) |
| E_ID3 Ex5 | 1/3 단위. TS − (PO+CF) = +0.67 | Ex1–Ex4 는 TS = PO+CF |
| E_ID4 Ex5 | 1/3 단위. TS − (PO+CF) = +0.67 | Ex1–Ex4 는 TS = PO+CF |

### 2.3 원저자 MATLAB 코드에서의 사용 [S]
- `data/raw` 안 모든 `.m` 파일에 Clinical 파일을 읽는 코드(`xlsread`, `readtable`, 파일명 참조)가 **없다**.
- 점수 관련 흔적은 `feat_extract_Ex5.m` 의 **주석 처리된** `Temporalscore5(…, 71, 92, 50, …)` 호출뿐이며, 이 함수 파일은 데이터에 없다 ([original_scripts.md](original_scripts.md) §2.1).
- 따라서 MATLAB 코드에서 이 값들이 어떻게 만들어지거나 쓰이는지 **확인할 수 없다**.

### 2.4 판정

| 대상 | 판정 |
|---|---|
| B_ID3, B_ID4, P_ID10, S_ID5, S_ID6 (불규칙 소수) | **원본 확인 필요** — 값 생성 방식 미확인. 오류로 단정하지 않는다 |
| E_ID3 Ex5, E_ID4 Ex5 (TS ≠ PO+CF) | **원본 확인 필요** — 어느 값이 기준인지 미확인. 값을 수정하지 않는다 |

- 처리 원칙: 이 값들은 수정·보정하지 않는다. 주 분석은 포함(A)으로 하고, 제외(B) 결과를 민감도 분석으로 함께 보고한다 (REQUIREMENTS §4.6, §12.4).
- TS ≠ PO+CF 2건은 PO·CF 를 target 으로 쓸 때는 영향이 없고, TS 를 쓰는 보조 분석에서만 표시한다.

---

## 3. Stroke 데이터 재검증 결과

결과 파일: `reports/gate3/label_availability_group.csv` (Subject 단위), `label_availability_exercise.csv` (Session 단위)

### 3.1 Group 별 Subject 수 [D]
각 열은 **Subject 수**다. PO/CF/TS 열 = 해당 점수가 1개 이상 있는 Subject.

| Group | Subject | Clinical | Score | PO | CF | TS |
|---|---:|---:|---:|---:|---:|---:|
| Expert | 17 | 15 | 14 | 14 | 14 | 14 |
| NotExpert | 27 | 25 | 25 | 25 | 25 | 25 |
| BackPain | 8 | 8 | 8 | 8 | 8 | 8 |
| Parkinson | 16 | 12 | 12 | 12 | 12 | 12 |
| Stroke | 10 | 5 | 5 | 5 | 5 | 5 |
| 합계 | 78 | 65 | 64 | 64 | 64 | 64 |

- Stroke: 폴더 10명 중 Clinical 파일이 있는 Subject 는 5명(S_ID2, S_ID3, S_ID5, S_ID6, S_ID7)이고, 5명 모두 15개 점수가 있다. 나머지 5명(S_ID1, S_ID4, S_ID8, S_ID9, S_ID10)은 다른 폴더를 포함해 Clinical 파일이 없다.
- 5명 중 2명(S_ID5, S_ID6)이 불규칙 소수 대상자다. 제외(민감도 B) 시 Stroke 점수 Subject 는 3명이다.

### 3.2 Stroke Exercise 별 Session 수 [D]
| Exercise | 세 점수 모두 있는 Session | 신호 QC 통과 + 세 점수 Session |
|---:|---:|---:|
| Ex1 | 5 | 0 |
| Ex2 | 5 | 0 |
| Ex3 | 5 | 0 |
| Ex4 | 5 | 0 |
| Ex5 | 5 | 0 |

- 현재 사본에서 Stroke 는 **신호와 점수가 함께 있는 Session 이 0개**다.
- 참고 (전체 Group 합): 신호 QC 통과 + 세 점수 Session 은 Ex1 6, Ex2 3, Ex3 7, Ex4 7, Ex5 6 이다.

### 3.3 Exercise 별 분석 가능 Subject [D]
결과 파일: `reports/gate3/analysis_feasibility.csv` (`python -m kimore.label_audit`)

**분석 가능 Subject** = 해당 Exercise 에서 신호 QC 를 통과하고(Position + Timestamp 파싱 성공, 녹화 1개) TS·PO·CF 가 모두 있는 Subject. 운동마다 Subject 1명 = Session 1개이므로 이 수는 Subject 수이자 Session 수다.

| 범위 | Ex1 | Ex2 | Ex3 | Ex4 | Ex5 |
|---|---:|---:|---:|---:|---:|
| 전체 | 6 | 3 | 7 | 7 | 6 |
| 건강군 | 2 | 2 | 6 | 5 | 4 |
| 환자군 | 4 | 1 | 1 | 2 | 2 |
| Expert | 1 | 0 | 1 | 3 | 1 |
| NotExpert | 1 | 2 | 5 | 2 | 3 |
| BackPain | 2 | 1 | 0 | 1 | 1 |
| Parkinson | 2 | 0 | 1 | 1 | 1 |
| Stroke | 0 | 0 | 0 | 0 | 0 |
| 전체 중 Age 있음 (Secondary) | 4 | 3 | 7 | 6 | 6 |
| 전체 중 불규칙 소수 제외 (민감도 B) | 6 | 3 | 7 | 6 | 6 |

- 가장 많은 운동도 7명이다. 예를 들어 5-fold CV 를 하면 fold 마다 test Subject 가 1–2명이다. 모델을 **만들 수 있는 것**과 그 결과를 **신뢰할 수 있는 것**은 다르므로, 현재 사본으로는 어떤 운동도 모델링하지 않는다.

---

## 4. Primary Analysis (사전 지정 초안) — Exercise 별 PO/CF 회귀

> 이 절은 **최종 확정이 아니다.** 원본 재확보 후 4.1 판정 규칙을 통과한 운동에 대해서만 사전 지정한다.

| 항목 | 내용 |
|---|---|
| 목적 | 운동 수행에서 추출한 운동학적 feature 가 임상 평가 점수(PO, CF)의 변이를 얼마나 설명하는지 평가한다 (RQ1, RQ2) |
| 표본 | Session (Subject × Exercise). 운동마다 Subject 1명 = 1관측 |
| Target | PO, CF 를 **각각** (target 별 별도 모델) |
| 모델 | Exercise 별 모델 (9절). 후보: Ridge, 트리 기반 1개 (REQUIREMENTS §11.4) |
| 비교 기준 | B1 전체 평균 · B2 Group 평균(참조용) · B3 Age-only · B4 feature 1–2개 선형 |
| 주 지표 | MAE (점수 단위). 보조: Spearman, RMSE, R² (단독 근거 아님) |
| 보고 | 전체 성능과 Group 내부 성능을 나눠 보고. 민감도 A/B(2절) 함께 보고 |
| 표현 원칙 | 진단·질병 판별 모델로 표현하지 않는다. 판별 성능을 목표로 두지 않는다. 임상적 효능을 주장하지 않는다 |

- 주의: 건강군 PO 상단이 압축되어 있을 가능성이 있다(만점 비율 34%). 오차를 Group 별·점수 구간별로 함께 확인한다.
- REQUIREMENTS §11.2 에 따라 이 내용은 **사전 지정 초안**이며, 원본 재확보 후 운동·target·feature set·모델·지표를 하나로 고정해 분석 전에 기록한다.

### 4.1 분석 수행 판정 규칙 (모든 분석에 적용)

> 분석 대상은 해당 Exercise 에서 **Signal 과 Clinical Score 가 모두 존재하는 Subject** 를 기준으로 결정한다.
> 최소 표본 수와 CV 가능 여부를 확인한 후 분석 수행 여부를 결정한다.
> 표본이 부족한 경우 해당 분석은 모델링 대상에서 제외하고 **기술통계 또는 탐색적 분석으로 제한**한다.

절차
1. 분석(Primary · Secondary · Exploratory · 민감도 B)과 범위(전체 · 코호트 · Group)마다 3.3 의 분석 가능 Subject 수 n 을 센다. Secondary 는 Age 가 있는 Subject 만 센다.
2. n 과 CV 조건을 아래 기준과 비교한다. 기준 값은 **결과를 보기 전에** 주 분석 사전 지정 문서에 숫자로 적는다 (현재 미정).
   - `N_min`: 모델링에 필요한 최소 분석 가능 Subject 수
   - fold 당 최소 test Subject 수, 층화 시 가장 작은 층의 최소 Subject 수 (11절)
   - 모델당 feature 수(6–10개 상한, REQUIREMENTS §6.3)에 비해 표본이 충분한지
3. 판정은 세 가지 중 하나다.

| 판정 | 조건 | 수행하는 것 |
|---|---|---|
| 모델링 | n ≥ N_min 이고 CV 조건 충족 | 사전 지정한 모델·CV·기준선 비교 |
| 기술통계·탐색만 | 0 < n < N_min 또는 CV 조건 미충족 | 분포·Subject 별 값 기술, 탐색적 상관. 모델 학습·성능 보고 없음 |
| 분석 불가 | n = 0 | 해당 범위는 결과 없음으로 보고하고 사유를 적는다 |

4. 판정 결과, n, 적용한 기준을 결과와 함께 보고한다. 기준을 바꾸면 사유와 날짜를 남긴다.

현재 사본 적용 결과
- n = 0 (분석 불가): Stroke 전 운동, Expert Ex2, Parkinson Ex2, BackPain Ex3.
- 그 외 범위: `N_min` 이 정해지지 않아 판정을 보류한다. 다만 전체 범위도 3–7명이라 어떤 합리적 기준에서도 모델링 판정을 받기 어렵다 [D].

**Stroke 를 모델에서 제외하는 이유 (보고용 문장)**
> Stroke 를 제외한 것은 해당 질환이 중요하지 않아서가 아니라, 현재 데이터에서 Signal 과 Clinical Score 가 동시에 존재하는 분석 가능 Subject 가 없어(0명) 모델 학습과 검증이 불가능했기 때문이다. 원본 재확보 후 같은 규칙으로 다시 판정한다.

## 5. Secondary Analysis — Age 를 공변량으로 포함한 회귀

| 항목 | 내용 |
|---|---|
| 목적 | Age 와 점수의 강한 연관을 고려할 때, 운동학적 feature 가 Age 만으로 설명되는 정보에 **추가** 설명력을 가지는지 확인한다 |
| 비교 | Age-only 모델 vs Age + feature 모델의 outer-CV MAE 차이 |
| 표본 | Age 와 점수가 모두 있는 Subject 의 Session (현재 56명) |
| 해석 | Age 의 인과 효과로 해석하지 않는다. Age 와 Group 이 강하게 얽혀 있어 회귀로도 완전히 분리되지 않을 수 있음을 함께 적는다 |

## 6. Exploratory Analysis — 환자군 내부 PO/CF 회귀

- 목적: 상한 근처 관측이 적은 환자군(GPP) 안에서 feature 와 점수의 관계를 탐색한다.
- 탐색 분석으로만 표기하고 결론의 근거로 쓰지 않는다.
- Stroke 는 점수 Subject 5명(제외 시 3명)이고 현재 신호+점수 Session 이 0개다. Stroke 결과는 **별도로 표시하고 단독 해석하지 않는다**.
- 환자군 안에서도 Age–점수 연관(ρ = −0.57)이 있으므로 Age-only 기준선을 함께 둔다.

## 6.1 TS 의 위치
- TS 는 대부분 PO + CF 와 같으므로(2절) **보조 분석으로만** 쓴다.
- TS 가 target 일 때 PO·CF 는 feature 로 쓰지 않는다 (target leakage, REQUIREMENTS §8.1).
- PO + CF 관계를 이유로 원본 TS·PO·CF 값을 다시 계산하거나 고치지 않는다.

## 6.2 이번 설계에 넣지 않은 것
- REQUIREMENTS §11.1 의 [선택] 보조 분석(건강군 vs 환자군 이진 분류)은 이번 설계에 포함하지 않았다. 수행 여부는 미정이다.

---

## 7. Age confounding 처리 원칙

| 구분 | 원칙 |
|---|---|
| 기본 | Age 를 중요한 confounder 후보로 취급한다. Age 분석 대상은 Age 가 실제로 있는 Subject 로 한정하고 그 n 을 함께 보고한다 |
| 주 분석과 함께 | Age 를 공변량으로 포함한 모델(5절)을 검토하고, Age-only 기준선(B3)을 모든 결과 옆에 둔다 |
| 민감도 분석 | 연령 매칭 subset 분석을 할 수 있다. 단 현재 ±5세 매칭 환자가 12명뿐이므로 **주 분석으로 쓰지 않는다** |
| 금지 | 결측 Age 를 보정·추정하지 않는다. Group 차이를 곧바로 운동 수행 능력 차이로 해석하지 않는다. Age 를 원인으로 서술하지 않는다 |

---

## 8. Session / Subject 정의와 분할 원칙

| 용어 | 정의 |
|---|---|
| Subject | 대상자 1명 (정규화된 Subject ID, 예: `S_ID5`) |
| Session | **Subject × Exercise** = ML 기본 표본 |
| Frame, Repetition | Session 내부 단위. 독립 표본이 아니다 |

**필수 원칙 — Subject-level split**
1. train/validation/test 분할은 Subject 단위로 한다. 같은 Subject 의 Ex1–Ex5 Session 은 모두 같은 쪽에 있어야 한다.
2. 분할에 쓰는 `groups` 인자는 항상 **SubjectID** 다.
3. Exercise 별 모델이라도 같은 `subject_id → fold` 파일을 공유한다. 한 Subject 는 모든 운동에서 같은 fold 에 속한다.
4. 모든 ML 코드는 학습 전에 `kimore.cv` 의 규칙 함수로 검사하고, 위반 시 중단한다.
   - `assert_valid_fold_assignment(assign, expected_subjects)`: Subject 중복·여러 fold 소속·누락 검사
   - `session_split_subjects(sessions, test_mask)` / `assert_no_subject_overlap(train, test)`: train/test Subject 겹침 검사
   - `forbidden_feature_columns(columns)`: Subject ID·Group·파일 정보·대리 변수·점수 계열 열이 feature 에 없는지 검사
5. 이 원칙은 `tests/test_cv.py`, `tests/test_gate3_design.py` 로 검증한다.

---

## 9. Exercise-specific model 원칙

- 기본값은 **Exercise 별 모델**이다. Ex1–Ex5 를 하나의 모델에 합치지 않는다.
  - 근거 [S]: 원저자 Script 의 P.O. 신호가 운동마다 다르다 (Ex1 어깨 기준 팔꿈치 각, Ex2 Hip 대비 Shoulder 기울기, Ex3 팔꿈치 간 x 거리, Ex4 SpineBase 이동, Ex5 무릎 분절각 합). 각 신호의 임상적 의미는 미확인 [E].
- feature dictionary, feature 계산, target, 분석, 모델을 운동별로 따로 정의한다.
- 통합 모델은 공통 feature 를 정의할 수 있고 데이터 규모가 충분하다고 판단될 때만 **탐색 분석**으로 검토한다. 그 경우에도 Subject-level split 을 지킨다.

---

## 10. Feature 설계 원칙 (구현하지 않음)

- 모델 하나당 feature 약 6–10개를 기본 상한으로 둔다. 많이 만든 뒤 고르는 방식은 쓰지 않는다 (REQUIREMENTS §6.3).
- 변수명만 보고 임상적 의미를 확정하지 않는다. 계산식은 원저자 Script 기준 [S], 의미는 확인 전까지 [E].
- 다음은 feature 로 쓰지 않는다: Subject ID, Group, 파일 정보, 프레임 수, 세션 길이, 녹화 날짜, tracking state 비율 (대리 변수, REQUIREMENTS §8.1).
- 각속도·반복 시간 등 시간이 필요한 feature 는 Timestamp 단위를 확인하기 전까지 계산하지 않는다.

| 후보 | 근거가 있는 운동 | 필요 데이터 | 상태 |
|---|---|---|---|
| ROM (반복별 max − min) | Ex1–Ex5 P.O. 신호에 피크 검출이 있음 [S] | Position, 반복 분할 | 계산 대상 신호는 [S], 임상적 의미 **confirmation needed** |
| 주요 관절/각도 | Ex1 팔꿈치 각(atan2), Ex2 몸통 기울기, Ex5 무릎 분절각 [S]; C.F. 의 팔꿈치·무릎·겨드랑이 각 [S] | Position | 계산식 [S], 어떤 동작을 나타내는지 **confirmation needed** |
| angular velocity | Script 에 없음 | Position, Timestamp | **confirmation needed** (Timestamp 단위 미확인) |
| repetition duration | Script 의 `der*` (최대 피크 간 샘플 수) [S] | 반복 분할, Timestamp | 샘플 수는 [S], 초 단위 변환 **confirmation needed** |
| repetition variability | `der*` 배열로 계산 가능 [S] | 반복 분할 | 반복 수가 적으면 불안정. **confirmation needed** |
| symmetry (좌우) | Ex1, Ex2, Ex5 는 좌·우 신호를 각각 계산 [S]. Ex3, Ex4 는 단일 신호 | Position | 운동별 정의 **confirmation needed** |
| compensation 관련 | C.F. 변수(예: `sommaaree`, `link_*`, `deltayknee`) [S] | Position | 변수명 해석은 [E]. **confirmation needed** |
| movement range (위치 이동 범위) | Ex4 SpineBase z·x 이동 [S], Ex3 팔꿈치 x 거리 [S] | Position | 축의 물리적 방향 미확인 [E]. **confirmation needed** |
| smoothness (SPARC, jerk) | Script 에 없음 | Position, Timestamp | [선택]. **confirmation needed** |

- 운동 Es1–Es5 의 실제 동작 이름은 Script 에 없다. 공식 문서로 확인하기 전에는 운동 특화 feature(예: 스쿼트 깊이)를 확정하지 않는다.

---

## 11. Subject-level CV 후보 (확정하지 않음)

모든 후보에서 `groups = subject_id` 다.

| 후보 | 장점 | 한계 (현재 데이터 기준 [D]) |
|---|---|---|
| StratifiedGroupKFold (층 = Group) | fold 마다 Group 비율을 비슷하게 유지. 작은 그룹이 한 fold 에 몰리는 것을 줄임 | 점수 Subject 가 Stroke 5명이라 k=5 이면 fold 당 Stroke 1명. Group 은 층화에만 쓰고 feature 로 쓰지 않음 |
| GroupKFold | 단순, Subject 겹침 없음 | Group 균형을 보장하지 않음. 작은 그룹이 일부 fold 에만 들어갈 수 있음 |
| Leave-One-Subject-Out (LeaveOneGroupOut, group = Subject) | 표본이 작을 때 학습 데이터를 최대로 사용 | 추정 분산이 큼. Nested CV 와 함께 쓰면 계산량이 큼 |

- 현재 사본은 운동별 신호+점수 Session 이 3–7개라 **어떤 방식도 실행할 수 없다**. 방식은 원본 재확보 후 운동별 실제 Subject 수와 Group 분포를 보고 정한다.
- 결정 기준(사전 기록): (1) 운동별 분석 가능 Subject 수, (2) 가장 작은 Group 의 Subject 수, (3) fold 당 test Subject 수, (4) 튜닝 시 Nested CV 가능 여부.
- 하이퍼파라미터 튜닝을 하면 Nested CV 를 쓴다 (REQUIREMENTS §9.1).

**fold assignment 저장 설계**
- 파일: `data/interim/folds/fold_assignment.csv` (Git 제외 경로) — 열: `subject_id, fold`
- 함께 기록: 분할 방식, k, random seed, 층화 변수, 기준 manifest 의 해시, 생성 일시 (`fold_assignment_meta.json`)
- 모든 실험이 같은 파일을 읽는다. 파일을 만들 때와 읽을 때 `kimore.cv.assert_valid_fold_assignment` 로 검사한다.
- 운동별로 분석 가능 Subject 가 다르면, 같은 파일에서 해당 운동의 Subject 만 골라 쓴다 (fold 를 다시 만들지 않는다).

---

## 12. 아직 확정하지 않은 사항

| 항목 | 상태 |
|---|---|
| 불규칙 소수 라벨 5명, TS ≠ PO+CF 2건 | 원본 확인 필요 |
| 주 분석의 최종 사전 지정 (운동, target, feature set, 모델, 지표 기준) | 원본 재확보 후 |
| CV 최종 방식과 k | 원본 재확보 후 (11절 기준) |
| 분석 수행 판정 기준 값 (`N_min`, fold 당 최소 test Subject 수 등) | 주 분석 사전 지정 시 숫자로 기록 (4.1) |
| Gate 번호 체계 (REQUIREMENTS §17 vs 13절 제안) | REQUIREMENTS 개정 시 결정 |
| Feature 최종 목록 | Gate 4(Script 해독 확정)·Gate 6 |
| Age 를 모델 입력으로 쓸지 (공변량 모델 외) | 미정 |
| Timestamp 단위, tracking state 값 의미 | 원본 문서 확인 필요 |
| 운동 Es1–Es5 의 실제 동작 정의 | 공식 문서 확인 필요 |
| Orientation 사용 여부 | 데이터 확보 후 |
| 필터·반복 분할 최종 파라미터 | 데이터 확보 후 (Gate 5) |
| 건강군 vs 환자군 이진 분류 보조 분석 수행 여부 | 미정 |
| 데이터 이용 조건·재배포 가능 여부 | 공식 배포처 확인 필요 |

---

## 13. 다음 단계 순서와 진입 조건

현재 사본으로는 Feature Extraction → ML 로 진행하지 않는다 (3.3, 4.1). 제약은 데이터 검증 단계에서 확인된 것이며, 아래 순서로 진행한다.

| 단계 (제안) | 내용 | 통과 조건 |
|---|---|---|
| **Gate 3.5 — 공식 원본 확보·비교** | 공식 원본과 현재 사본을 둘 다 READ-ONLY 로 비교. Session·Signal·Clinical·SuppInfo 재집계. 2절 라벨 7건 확인. Exercise 별 분석 가능 Subject 수 확인 | 비교 결과 기록, `kimore.qc`·`eda`·`label_eda`·`label_audit` 재실행, 4.1 규칙으로 운동별 판정 |
| **Gate 4 — Feature Extraction 설계** | 원저자 MATLAB 의 P.O./C.F. 의미 확인, 좌표축 확인, Timestamp 단위 확인, 반복 동작 구간 확인, feature 정의 | 10절 **confirmation needed** 해소, 운동별 feature dictionary 초안 |
| **Gate 5 — Feature Extraction 구현** | Python 으로 원저자 MATLAB 계산 재현, 동일 입력에서 결과 비교, Feature QC | 재현 오차 기록, 실패 Session 기록 |
| **Gate 6 — ML** | Subject-level CV, Baseline → Ridge → Tree, MAE 중심 평가 | 4.1 에서 '모델링' 판정을 받은 분석만 수행 |

> **REQUIREMENTS 와의 차이 (확정하지 않고 보고)**: REQUIREMENTS §17 의 Gate 번호는 4 = 원저자 Script 해독·재현, 5 = 전처리·반복 분할, 6 = Feature engineering, 7 = 통계 분석, 8 = 모델링이다. 위 표의 번호(3.5 / 4 / 5 / 6)는 이와 다르다. 내용 순서는 같은 방향(원본 확보 → Script 해독 → 전처리·feature → 모델링)이지만, 어느 번호 체계를 쓸지는 REQUIREMENTS 를 고칠 때 결정한다. 그 전까지 REQUIREMENTS 의 번호를 공식 번호로 둔다.

### Gate 3.5 진입 전 체크리스트
1. **원본 경로**: 공식 원본의 위치를 정하고 현재 사본(`data/raw`)과 따로 둔다. 둘 다 읽기만 한다.
2. **라벨 확인**: 2절의 "원본 확인 필요" 7건을 공식 원본에서 다시 확인하고 결과를 이 문서에 기록한다.
3. **표본 확인**: 3.3 표를 공식 원본으로 다시 만들고 4.1 규칙으로 운동·범위별 판정을 기록한다. 특히 Stroke.
4. **CV 확정**: '모델링' 판정을 받은 분석에 대해 11절 기준으로 방식을 정하고 fold assignment 파일을 저장한다. `tests/test_cv.py`, `tests/test_gate3_design.py` 통과.
5. **주 분석 사전 지정**: 4절 초안과 4.1 의 `N_min` 등 기준 값을 분석 전에 docs/ 에 숫자로 기록한다.
6. **운동 정의·Script 의미 확인**: Es1–Es5 동작과 P.O./C.F. 신호의 의미를 공식 문서로 확인한다 (Gate 4 입력).
