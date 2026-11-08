"""Gate 3.5 — 두 KIMORE 데이터 트리(현재 사본 vs 공식 원본)를 READ-ONLY 로 비교한다.

사용:
    python -m kimore.compare_sources --current data/raw --official <공식 원본 경로> --out reports/gate3_5

산출물 (out 폴더, CSV)
- inventory_comparison.csv          Group × Subject × Exercise: Raw/Label/Script 폴더, 파일 종류별 존재
- file_comparison.csv               상대경로별 존재·크기·SHA-256 (영상은 해시 비교 제외)
- session_comparison.csv            Subject × Exercise: 파일 존재, 점수 존재, qc_status, 분석 가능 여부
- clinical_comparison.csv           Subject × Exercise: TS/PO/CF, 결측, TS−(PO+CF), 값 형식
- signal_comparison.csv             Session: 행 수, frame 수, Timestamp 간격, 구조(state 값, Orientation 0값 관절)
- analysis_feasibility_comparison.csv  Exercise × 범위: 분석 가능 Subject 수 (현재 vs 공식)

원칙
- 두 경로 모두 읽기만 한다. 어떤 파일도 복사·이동·수정하지 않는다.
- kimore.qc / kimore.labels / kimore.label_audit 의 기존 규칙을 그대로 써서 두 트리를 같은 방식으로 집계한다.
- 차이는 값 그대로 기록하고 원인을 추정하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kimore import label_audit, labels, qc

SIDES = ("current", "official")
NO_HASH_KINDS = ("video", "shortcut")
MARKER = ".generated_by_kimore_compare_sources"


# ------------------------------------------------------------------ build one side


def build_side(raw_root: str | Path) -> dict[str, pd.DataFrame]:
    """한 트리를 kimore.qc 와 같은 규칙으로 집계한다 (파일을 쓰지 않음)."""
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    files = qc.scan_files(raw_root, compute_hash=True)
    subjects, wide, _ = qc.build_subjects(raw_root, files)
    sessions = qc.build_sessions(raw_root, files, subjects)
    long = labels.clinical_long(wide)
    return {"files": files, "subjects": subjects, "sessions": sessions, "long": long}


# ------------------------------------------------------------------ comparisons


def _both(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """keys 로 outer join 하고 cols 마다 _current/_official 와 same_<col> 을 만든다."""
    a = a[keys + cols].assign(in_current=True)
    b = b[keys + cols].assign(in_official=True)
    m = a.merge(b, on=keys, how="outer", suffixes=("_current", "_official"))
    m["in_current"] = m["in_current"].fillna(False).astype(bool)
    m["in_official"] = m["in_official"].fillna(False).astype(bool)
    for c in cols:
        x, y = m[f"{c}_current"], m[f"{c}_official"]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            same = np.isclose(x.astype(float), y.astype(float), rtol=0, atol=0, equal_nan=True)
        else:
            same = (x.astype(str) == y.astype(str)) | (x.isna() & y.isna())
        m[f"same_{c}"] = same & m["in_current"] & m["in_official"]
    return m


def compare_files(cur: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    cols = ["group", "subject_folder", "exercise_folder", "subfolder", "kind", "is_copy_name", "size_bytes", "sha256"]
    m = _both(cur, off, ["path"], cols)
    kind = m["kind_current"].fillna(m["kind_official"])
    m["kind"] = kind
    m["hash_compared"] = ~kind.isin(NO_HASH_KINDS) & m["in_current"] & m["in_official"]
    m["status"] = np.select(
        [~m["in_official"], ~m["in_current"], ~m["hash_compared"], m["same_sha256"]],
        ["current_only", "official_only", "both_not_hashed", "identical"],
        default="content_differs",
    )
    return m.sort_values("path").reset_index(drop=True)


def _inventory(files: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    grid = pd.DataFrame([(s, g, e) for s, g in zip(subjects["subject_id"], subjects["group"])
                         for e in qc.EXERCISE_DIRS], columns=["subject_id", "group", "exercise_folder"])
    f = files[files["exercise_folder"].notna()]
    out = grid.copy()
    for sub in ("Raw", "Label", "Script"):
        has = f[f["subfolder"] == sub].groupby(["subject_folder", "exercise_folder"]).size()
        out[f"has_{sub}"] = [has.get((s, e), 0) > 0 for s, e in zip(out["subject_id"], out["exercise_folder"])]
    for kind in ("position", "orientation", "timestamp", "clinical", "suppinfo"):
        has = f[f["kind"] == kind].groupby(["subject_folder", "exercise_folder"]).size()
        out[f"n_{kind}_files"] = [int(has.get((s, e), 0)) for s, e in zip(out["subject_id"], out["exercise_folder"])]
    return out


def compare_inventory(c: dict, o: dict) -> pd.DataFrame:
    a, b = _inventory(c["files"], c["subjects"]), _inventory(o["files"], o["subjects"])
    cols = [x for x in a.columns if x.startswith(("has_", "n_"))]
    return _both(a, b, ["group", "subject_id", "exercise_folder"], cols)


SESSION_COLS = ["session_folder_exists", "has_position", "has_orientation", "has_timestamp", "has_clinical",
                "has_suppinfo", "has_scores", "n_recordings", "qc_status", "qc_reasons", "analyzable_signal",
                "analyzable_with_scores"]
SIGNAL_COLS = ["position_n_rows_raw", "orientation_n_rows_raw", "timestamp_n_rows_raw", "position_n_frames",
               "orientation_n_frames", "timestamp_n_frames", "n_frames", "ts_median_interval_raw", "ts_n_gap",
               "ts_n_duplicate", "ts_n_backward", "state_values", "state1_ratio", "ori_all_zero_joints",
               "ori_zero_matches_script"]


def compare_sessions(c: dict, o: dict) -> pd.DataFrame:
    return _both(c["sessions"].reindex(columns=["subject_id", "group", "exercise", *SESSION_COLS]),
                 o["sessions"].reindex(columns=["subject_id", "group", "exercise", *SESSION_COLS]),
                 ["subject_id", "group", "exercise"], SESSION_COLS)


def compare_signals(c: dict, o: dict) -> pd.DataFrame:
    return _both(c["sessions"].reindex(columns=["subject_id", "group", "exercise", *SIGNAL_COLS]),
                 o["sessions"].reindex(columns=["subject_id", "group", "exercise", *SIGNAL_COLS]),
                 ["subject_id", "group", "exercise"], SIGNAL_COLS)


def _clinical_table(side: dict) -> pd.DataFrame:
    subj = side["subjects"][["subject_id", "group", "C1_has_clinical_file", "value_format"]]
    d = side["long"].merge(subj, on="subject_id", how="right")
    d["n_missing"] = d[list(labels.SCORES)].isna().sum(axis=1)
    d["delta"] = d["TS"] - (d["PO"] + d["CF"])
    for s in labels.SCORES:
        d[f"{s}_format"] = [labels.classify_value(v) for v in d[s]]
    return d


def compare_clinical(c: dict, o: dict, extra_targets: list[tuple[str, int | None]] = ()) -> pd.DataFrame:
    cols = ["C1_has_clinical_file", "value_format", *labels.SCORES, "n_missing", "delta",
            *[f"{s}_format" for s in labels.SCORES]]
    m = _both(_clinical_table(c), _clinical_table(o), ["subject_id", "group", "exercise"], cols)
    m["same_scores"] = m[[f"same_{s}" for s in labels.SCORES]].all(axis=1)
    # 재검증 대상: 어느 쪽에서든 불규칙 소수 / TS≠PO+CF 로 잡힌 것 + 지정 대상
    flag = (m["value_format_current"].eq("irregular") | m["value_format_official"].eq("irregular")
            | (m["delta_current"].abs() > labels.SUM_TOL) | (m["delta_official"].abs() > labels.SUM_TOL))
    for sid, ex in extra_targets:
        flag |= (m["subject_id"] == sid) & ((m["exercise"] == ex) if ex is not None else True)
    m["recheck_target"] = flag
    return m.sort_values(["group", "subject_id", "exercise"]).reset_index(drop=True)


def compare_feasibility(c: dict, o: dict) -> pd.DataFrame:
    fa = label_audit.analysis_feasibility(c["sessions"], c["long"], c["subjects"])
    fb = label_audit.analysis_feasibility(o["sessions"], o["long"], o["subjects"])
    cols = ["n_subjects_in_scope", "n_analyzable_subjects", "n_analyzable_with_age",
            "n_analyzable_excl_irregular", "status"]
    m = _both(fa, fb, ["exercise", "scope"], cols)
    m["diff_analyzable"] = m["n_analyzable_subjects_official"] - m["n_analyzable_subjects_current"]
    return m


# ------------------------------------------------------------------ run


GATE3_TARGETS: list[tuple[str, int | None]] = [
    ("B_ID3", None), ("B_ID4", None), ("P_ID10", None), ("S_ID5", None), ("S_ID6", None),
    ("E_ID3", 5), ("E_ID4", 5),
]


def _prepare_out(path: Path) -> None:
    if path.exists() and any(path.iterdir()) and not (path / MARKER).exists():
        raise FileExistsError(f"{path} 에 이 도구가 만들지 않은 파일이 있습니다. 다른 --out 을 지정하세요.")
    path.mkdir(parents=True, exist_ok=True)
    (path / MARKER).write_text("kimore.compare_sources 가 생성·갱신하는 폴더입니다.\n", encoding="utf-8")


def run(current: str | Path, official: str | Path, out: str | Path) -> dict[str, pd.DataFrame]:
    current, official, out = Path(current).resolve(), Path(official).resolve(), Path(out).resolve()
    if current == official:
        raise ValueError("현재 사본과 공식 원본 경로가 같습니다.")
    for root in (current, official):
        if out == root or root in out.parents:
            raise ValueError(f"출력 폴더가 원본 트리 안에 있습니다: {out}")
    c, o = build_side(current), build_side(official)
    res = {
        "inventory_comparison": compare_inventory(c, o),
        "file_comparison": compare_files(c["files"], o["files"]),
        "session_comparison": compare_sessions(c, o),
        "clinical_comparison": compare_clinical(c, o, GATE3_TARGETS),
        "signal_comparison": compare_signals(c, o),
        "analysis_feasibility_comparison": compare_feasibility(c, o),
    }
    _prepare_out(out)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    return res


def summary(res: dict[str, pd.DataFrame]) -> dict:
    f = res["file_comparison"]
    s = res["session_comparison"]
    cl = res["clinical_comparison"]
    return {
        "files_by_status": f["status"].value_counts().to_dict(),
        "sessions_with_any_difference": int((~s[[c for c in s.columns if c.startswith("same_")]].all(axis=1)).sum()),
        "clinical_rows_with_score_difference": int((~cl["same_scores"]).sum()),
        "feasibility_changed_cells": int((res["analysis_feasibility_comparison"]["diff_analyzable"].fillna(1) != 0).sum()),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="현재 사본 vs 공식 원본 READ-ONLY 비교 (Gate 3.5)")
    ap.add_argument("--current", default="data/raw")
    ap.add_argument("--official", required=True)
    ap.add_argument("--out", default="reports/gate3_5")
    args = ap.parse_args(argv)
    res = run(args.current, args.official, args.out)
    print(summary(res))


if __name__ == "__main__":
    main()
