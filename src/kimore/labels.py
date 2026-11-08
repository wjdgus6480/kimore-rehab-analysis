"""ClinicalAssessment / SuppInfo 로딩과 Label QC.

근거 (REQUIREMENTS §4.5, §4.6)
- 점수 파일: 1행, 'Subject ID' + 'clinical TS/PO/CF Ex#1~5' 15열 [D]
- 범위: TS 0~50, PO 0~15, CF 0~35
- TS = PO + CF는 "대부분 성립"이므로 불일치는 오류로 단정하지 않고 QC 항목으로만 기록한다.
- 결측 값은 보정·추정하지 않는다.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from kimore.io import ParseError

SCORES: tuple[str, ...] = ("TS", "PO", "CF")
EXERCISES: tuple[int, ...] = (1, 2, 3, 4, 5)
RANGES: dict[str, tuple[float, float]] = {"TS": (0, 50), "PO": (0, 15), "CF": (0, 35)}
SUM_TOL = 0.01

QC_COLUMNS = ["subject_id", "exercise", "check", "detail"]

_CLIN_HEADER_RE = re.compile(r"^\s*clinical\s+(TS|PO|CF)\s+Ex\s*#?\s*(\d)\s*$", re.IGNORECASE)
_SUBJECT_HEADER_RE = re.compile(r"^\s*subject\s*id\s*$", re.IGNORECASE)


def score_columns() -> list[str]:
    """['TS_Ex1'..'TS_Ex5', 'PO_Ex1'.., 'CF_Ex1'..] 순서."""
    return [f"{s}_Ex{e}" for s in SCORES for e in EXERCISES]


def _to_float(v) -> float:
    if v is None:
        return math.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


def _to_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def _read_sheet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_excel(path, header=0, engine="openpyxl")
    except Exception as exc:  # 손상·비xlsx 파일
        raise ParseError(f"{path.name}: xlsx를 읽을 수 없습니다 ({exc})") from exc


def read_clinical(path: str | Path) -> dict:
    """ClinicalAssessment xlsx → {'subject_in_file', 'TS_Ex1', ...}. 없는 값은 nan."""
    path = Path(path)
    df = _read_sheet(path)

    col_map: dict[str, str] = {}
    subject_col = None
    for col in df.columns:
        name = str(col)
        if _SUBJECT_HEADER_RE.match(name):
            subject_col = col
            continue
        m = _CLIN_HEADER_RE.match(name)
        if m:
            col_map[f"{m.group(1).upper()}_Ex{int(m.group(2))}"] = col

    missing = [c for c in score_columns() if c not in col_map]
    if missing:
        raise ParseError(f"{path.name}: 점수 헤더 누락 {missing}")

    row = df.iloc[0] if len(df) else None
    out: dict = {"subject_in_file": _to_str(row[subject_col]) if (row is not None and subject_col is not None) else None}
    for c in score_columns():
        out[c] = _to_float(row[col_map[c]]) if row is not None else math.nan
    return out


def read_suppinfo(path: str | Path) -> dict:
    """SuppInfo xlsx → {'subject_in_file', 'group', 'age', 'gender'}."""
    path = Path(path)
    df = _read_sheet(path)
    norm = {re.sub(r"\s+", "", str(c)).lower(): c for c in df.columns}
    needed = {"subjectid": "subject_in_file", "group": "group", "age": "age", "gender": "gender"}
    absent = [k for k in needed if k not in norm]
    if absent:
        raise ParseError(f"{path.name}: SuppInfo 헤더 누락 {absent}")

    row = df.iloc[0] if len(df) else None

    def get(key):
        return row[norm[key]] if row is not None else None

    return {
        "subject_in_file": _to_str(get("subjectid")),
        "group": _to_str(get("group")),
        "age": _to_float(get("age")),
        "gender": _to_str(get("gender")),
    }


def classify_value(v) -> str:
    """'integer' | 'third' | 'irregular' | 'missing'."""
    f = _to_float(v)
    if math.isnan(f):
        return "missing"
    if abs(f - round(f)) < 1e-6:
        return "integer"
    if abs(3 * f - round(3 * f)) < 1e-6:
        return "third"
    return "irregular"


def subject_value_format(values: Iterable) -> str:
    """한 대상자 점수들의 형식. irregular > third > integer > missing 우선."""
    kinds = {classify_value(v) for v in values}
    for k in ("irregular", "third", "integer"):
        if k in kinds:
            return k
    return "missing"


def label_qc(wide: pd.DataFrame) -> pd.DataFrame:
    """결측·범위 이탈·TS≠PO+CF 탐지. index=subject_id."""
    issues = []
    for sid, row in wide.iterrows():
        for e in EXERCISES:
            vals = {s: _to_float(row.get(f"{s}_Ex{e}")) for s in SCORES}
            for s, v in vals.items():
                if math.isnan(v):
                    issues.append((sid, e, "missing", f"{s}_Ex{e} 결측"))
                    continue
                lo, hi = RANGES[s]
                if not (lo <= v <= hi):
                    issues.append((sid, e, "out_of_range", f"{s}_Ex{e}={v} (범위 {lo}~{hi})"))
            if not any(math.isnan(v) for v in vals.values()):
                diff = vals["TS"] - (vals["PO"] + vals["CF"])
                if abs(diff) > SUM_TOL:
                    issues.append((sid, e, "sum_mismatch", f"TS-(PO+CF)={diff:.4f}"))
    out = pd.DataFrame(issues, columns=QC_COLUMNS)
    if not out.empty:
        out["exercise"] = out["exercise"].astype(int)
    return out


def clinical_long(wide: pd.DataFrame) -> pd.DataFrame:
    """wide(index=subject_id) → [subject_id, exercise, TS, PO, CF]."""
    rows = []
    for sid, row in wide.iterrows():
        for e in EXERCISES:
            rows.append(
                {
                    "subject_id": sid,
                    "exercise": e,
                    **{s: _to_float(row.get(f"{s}_Ex{e}")) for s in SCORES},
                }
            )
    return pd.DataFrame(rows, columns=["subject_id", "exercise", *SCORES])
