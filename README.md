# KIMORE 재활운동 수행 정량화 프로젝트

Kinect v2 기반 재활운동 데이터(KIMORE)에서 움직임 특성을 정량화하고,
그 지표와 임상평가 점수(PO/CF)의 관계 및 예측 가능성을 검증하는 프로젝트입니다.
질병을 진단하거나 판별하는 시스템이 아닙니다.

- 요구사항: [REQUIREMENTS.md](REQUIREMENTS.md)
- 현재 단계: Gate 1 (인벤토리·파싱·QC). 로컬 데이터가 불완전하므로 결과는 모두 **개발용·예비** 결과입니다.

## 폴더 구조

```
data/raw/        KIMORE 원본 (읽기 전용, Git 제외)
data/interim/    manifest, 라벨 테이블 등 중간 산출물 (Git 제외)
data/processed/  Session 단위 분석 테이블 (Git 제외)
src/kimore/      io.py(파싱·schema), labels.py(라벨 로딩·QC), qc.py(인벤토리·manifest),
                 eda.py(EDA 테이블·통계), eda_plots.py(그래프·EDA 보고서)
reports/eda/     EDA 그래프·통계표·보고서 (kimore.eda 가 생성)
tests/           구조 검증 테스트 (수치 하드코딩 없음)
docs/            인벤토리 보고서, 결정 기록
notebooks/       탐색 분석
models/          저장된 모델
```

## 환경 설정 (Windows)

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 실행

```bash
# 인벤토리·manifest·라벨 QC 생성 → data/interim/, docs/inventory_report.md
PYTHONPATH=src .venv/Scripts/python.exe -m kimore.qc --raw data/raw --out data/interim --report docs/inventory_report.md

# EDA·데이터 품질 시각화 (kimore.qc 산출물 재사용, 원본은 읽기만) → data/interim/eda/, reports/eda/
PYTHONPATH=src .venv/Scripts/python.exe -m kimore.eda --raw data/raw --interim data/interim --out reports/eda

# Gate 2 라벨·Age EDA (원본은 읽기만) → docs/label_eda_report.md, docs/figures/label_eda/
PYTHONPATH=src .venv/Scripts/python.exe -m kimore.label_eda

# Gate 3 진입 전 라벨 재검증·보유 재집계 (원본은 읽기만) → reports/gate3/  (설계: docs/gate3_analysis_design.md)
PYTHONPATH=src .venv/Scripts/python.exe -m kimore.label_audit

# Gate 3.5 현재 사본 vs 공식 원본 비교 (두 경로 모두 읽기만) → reports/gate3_5/
PYTHONPATH=src .venv/Scripts/python.exe -m kimore.compare_sources --current data/raw --official <공식 원본 경로>

# 구조 테스트
.venv/Scripts/python.exe -m pytest
```

(`src`를 import 경로에 넣기 위해 `PYTHONPATH=src`가 필요하며, pytest는 `pytest.ini`에서 자동 설정됩니다.)

### 이 저장소에 포함하지 않는 것
- KIMORE 원본 데이터(`data/raw/`)와 중간 산출물(`data/interim/`, `data/processed/`)
- 데이터에서 계산된 결과물: `reports/`, `docs/inventory_report.md`, `docs/label_eda_report.md`, `docs/figures/`
  — Subject 단위 점수·나이 등이 들어 있어, 데이터 이용 조건을 확인하기 전에는 공개하지 않습니다. 원본을 받은 뒤 위 명령으로 로컬에서 생성됩니다.
- 설계 문서(`docs/gate3_analysis_design.md` 등)에는 집계 수치와 가명 Subject ID만 있고 개별 점수 값은 없습니다.

## 데이터 출처와 인용

- 데이터셋: KIMORE — [UNIVPM VRAI 배포 페이지](https://vrai.dii.univpm.it/content/kimore-dataset)
- 인용: M. Capecci et al., "The KIMORE Dataset: KInematic Assessment of MOvement and Clinical Scores
  for Remote Monitoring of Physical REhabilitation," *IEEE Transactions on Neural Systems and
  Rehabilitation Engineering*, vol. 27, no. 7, pp. 1436–1448, 2019. DOI: 10.1109/TNSRE.2019.2923060
- 이용 조건·재배포 가능 여부: **공식 배포처의 이용 조건 확인 필요** (확인 전 데이터 외부 업로드·재배포 금지)
