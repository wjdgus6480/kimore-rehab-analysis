"""Gate 3 진입 전 라벨 재검증과 라벨 보유 재집계 (원본 읽기 전용).

사용:
    python -m kimore.label_audit --raw data/raw --interim data/interim --out reports/gate3

산출물 (out 폴더, CSV)
- label_recheck_files.csv     대상 Subject 의 Clinical 파일별: 위치·해시·복사본 여부, 셀 형식, 수식 개수, 파일 메타데이터
- label_recheck_values.csv    대상 Subject × Exercise: TS/PO/CF 원본 값, 값 형식, TS−(PO+CF)
- label_availability_group.csv  Group 별 Subject 수: 전체 / Clinical 파일 / 점수 1개 이상 / PO / CF / TS 보유
- label_availability_exercise.csv  Group × Exercise 별 Session 수: 세 점수 모두 / 신호 QC 통과 + 세 점수
- analysis_feasibility.csv     Exercise × 범위(전체·코호트·Group)별 분석 가능 Subject 수와 판정

원칙
- 원본은 openpyxl 로 읽기만 한다. 값을 고치거나 원인을 추정하지 않는다.
- 'Subject 수'와 'Session 수'를 다른 표로 분리한다.
- 수치는 실행할 때마다 계산한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
import pandas as pd

from kimore import labels
from kimore.qc import GROUPS

GROUP_ORDER: tuple[str, ...] = tuple(GROUPS)


def flagged_subjects(subjects: pd.DataFrame, long: pd.DataFrame) -> dict[str, list]:
    """재검증 대상: 불규칙 소수 대상자와 TS≠PO+CF Session (데이터에서 계산)."""
    irregular = sorted(subjects.loc[subjects["value_format"] == "irregular", "subject_id"])
    d = long.dropna(subset=list(labels.SCORES))
    mism = d[(d["TS"] - (d["PO"] + d["CF"])).abs() > labels.SUM_TOL]
    return {"irregular": irregular, "sum_mismatch": sorted(zip(mism["subject_id"], mism["exercise"].astype(int)))}


def recheck_files(raw_root: str | Path, files: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    """대상 Subject 의 모든 Clinical 파일(복사본·다른 폴더 포함)을 셀 단위로 다시 읽는다."""
    raw_root = Path(raw_root)
    clin = files[(files["kind"] == "clinical") & files["label_subject_id"].isin(subject_ids)]
    rows = []
    for r in clin.sort_values(["label_subject_id", "path"]).itertuples():
        wb = openpyxl.load_workbook(raw_root / r.path, read_only=False, data_only=False)
        ws = wb.worksheets[0]
        score_cells = [ws.cell(2, j) for j in range(2, 2 + len(labels.score_columns()))]
        rows.append({
            "subject_id": r.label_subject_id,
            "path": r.path,
            "is_copy_name": bool(r.is_copy_name),
            "misplaced": bool(r.misplaced),
            "sha256": r.sha256,
            "n_sheets": len(wb.worksheets),
            "n_formula_cells": sum(isinstance(c.value, str) and c.value.startswith("=") for c in score_cells),
            "number_formats": "|".join(sorted({c.number_format for c in score_cells if c.value is not None})),
            "creator": wb.properties.creator,
            "last_modified_by": wb.properties.lastModifiedBy,
            "created": wb.properties.created,
            "modified": wb.properties.modified,
        })
        wb.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out["n_distinct_contents_for_subject"] = out.groupby("subject_id")["sha256"].transform("nunique")
    return out


def recheck_values(long: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    d = long[long["subject_id"].isin(subject_ids)].copy()
    d["delta_TS_minus_PO_CF"] = d["TS"] - (d["PO"] + d["CF"])
    for s in labels.SCORES:
        d[f"{s}_format"] = [labels.classify_value(v) for v in d[s]]
    return d.sort_values(["subject_id", "exercise"]).reset_index(drop=True)


def availability_by_group(subjects: pd.DataFrame) -> pd.DataFrame:
    """Subject 단위 표. 각 열은 '해당 점수가 1개 이상 있는 Subject 수'다."""
    s = subjects.copy()
    for sc in labels.SCORES:
        cols = [f"{sc}_Ex{e}" for e in labels.EXERCISES]
        s[f"has_{sc}"] = s.reindex(columns=cols).notna().any(axis=1)
    t = s.groupby("group").agg(
        Subject=("subject_id", "size"),
        Clinical=("C1_has_clinical_file", "sum"),
        Score=("C2_has_any_score", "sum"),
        PO=("has_PO", "sum"),
        CF=("has_CF", "sum"),
        TS=("has_TS", "sum"),
    ).reindex([g for g in GROUP_ORDER if g in set(s["group"])]).astype(int)
    t.loc["합계"] = t.sum()
    t.index.name = "Group"
    return t.reset_index()


def availability_by_exercise(sessions: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Session 단위 표 (Group × Exercise)."""
    d = sessions[["subject_id", "group", "exercise", "analyzable_signal"]].merge(
        long, on=["subject_id", "exercise"], how="left")
    d["all_scores"] = d[list(labels.SCORES)].notna().all(axis=1)
    d["signal_and_scores"] = d["all_scores"] & d["analyzable_signal"].astype(bool)
    t = d.groupby(["group", "exercise"]).agg(
        sessions_in_grid=("subject_id", "size"),
        sessions_all_scores=("all_scores", "sum"),
        sessions_signal_and_scores=("signal_and_scores", "sum"),
    ).reset_index()
    t["group"] = pd.Categorical(t["group"], categories=GROUP_ORDER, ordered=True)
    return t.sort_values(["group", "exercise"]).reset_index(drop=True)


FEASIBILITY_PENDING = "기준 확정 후 판정"
FEASIBILITY_NONE = "모델링 불가 (분석 가능 Subject 0명)"


def analysis_feasibility(sessions: pd.DataFrame, long: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    """Exercise × 범위(전체·코호트·Group)별 '분석 가능 Subject' 수 (docs/gate3_analysis_design.md §4.1).

    분석 가능 Subject = 해당 Exercise 에서 신호 QC 를 통과하고(analyzable_signal) TS·PO·CF 가 모두 있는 Subject.
    운동마다 Subject 1명 = Session 1개이므로 이 수는 Subject 수이자 Session 수다.
    최소 표본 기준은 아직 정하지 않았으므로, 0명인 경우만 '모델링 불가'로 판정하고 나머지는 보류한다.
    """
    meta = subjects[["subject_id", "group_top", "age", "value_format"]]
    d = sessions[["subject_id", "group", "exercise", "analyzable_signal"]].merge(
        long, on=["subject_id", "exercise"], how="left").merge(meta, on="subject_id", how="left")
    d["ok"] = d["analyzable_signal"].astype(bool) & d[list(labels.SCORES)].notna().all(axis=1)
    d["cohort"] = d["group_top"].map({"CG": "건강군", "GPP": "환자군"})
    scopes = [("전체", None, None), ("건강군", "cohort", "건강군"), ("환자군", "cohort", "환자군")]
    scopes += [(g, "group", g) for g in GROUP_ORDER]
    rows = []
    for ex in labels.EXERCISES:
        e = d[d["exercise"] == ex]
        for name, col, val in scopes:
            s = e if col is None else e[e[col] == val]
            ok = s[s["ok"]]
            n = int(ok["subject_id"].nunique())
            rows.append({
                "exercise": ex, "scope": name,
                "n_subjects_in_scope": int(s["subject_id"].nunique()),
                "n_analyzable_subjects": n,
                "n_analyzable_with_age": int(ok.loc[ok["age"].notna(), "subject_id"].nunique()),
                "n_analyzable_excl_irregular": int(ok.loc[ok["value_format"] != "irregular", "subject_id"].nunique()),
                "status": FEASIBILITY_NONE if n == 0 else FEASIBILITY_PENDING,
            })
    return pd.DataFrame(rows)


def run(raw_root, interim, out) -> dict[str, pd.DataFrame]:
    interim, out = Path(interim), Path(out)
    files = pd.read_csv(interim / "manifest_files.csv", encoding="utf-8-sig")
    subjects = pd.read_csv(interim / "subjects.csv", encoding="utf-8-sig")
    sessions = pd.read_csv(interim / "manifest_sessions.csv", encoding="utf-8-sig")
    long = pd.read_csv(interim / "labels_clinical_long.csv", encoding="utf-8-sig")

    flags = flagged_subjects(subjects, long)
    targets = sorted(set(flags["irregular"]) | {sid for sid, _ in flags["sum_mismatch"]})
    res = {
        "label_recheck_files": recheck_files(raw_root, files, targets),
        "label_recheck_values": recheck_values(long, targets),
        "label_availability_group": availability_by_group(subjects),
        "label_availability_exercise": availability_by_exercise(sessions, long),
        "analysis_feasibility": analysis_feasibility(sessions, long, subjects),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    res["flags"] = flags
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="라벨 재검증·보유 재집계 (원본은 읽기만 함)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--out", default="reports/gate3")
    args = ap.parse_args(argv)
    res = run(args.raw, args.interim, args.out)
    f = res["flags"]
    print(f"irregular={f['irregular']} sum_mismatch={f['sum_mismatch']} "
          f"files={len(res['label_recheck_files'])}")


if __name__ == "__main__":
    main()
