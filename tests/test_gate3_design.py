"""Gate 3 진입 전 구조 테스트 (docs/gate3_analysis_design.md §8, REQUIREMENTS §14).

- 수치를 하드코딩하지 않고 불변 조건만 검사한다.
- 실제 data/raw, data/interim 이 없으면 해당 테스트는 skip 한다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kimore import cv, label_audit, labels, qc
from test_qc import raw  # noqa: F401  (합성 원본 fixture 재사용)

ROOT = Path(__file__).resolve().parents[1]
REAL_RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
needs_interim = pytest.mark.skipif(not (INTERIM / "manifest_sessions.csv").exists(), reason="data/interim 없음")
needs_raw = pytest.mark.skipif(not REAL_RAW.is_dir(), reason="data/raw 없음")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(INTERIM / name, encoding="utf-8-sig")


def _subject_folds(subject_ids, k: int, seed: int = 0) -> pd.DataFrame:
    """테스트용 subject → fold 배정 (실제 CV 방식이 아니다)."""
    ids = np.array(sorted(set(subject_ids)))
    np.random.default_rng(seed).shuffle(ids)
    return pd.DataFrame({"subject_id": ids, "fold": np.arange(len(ids)) % k})


# ------------------------------------------------- 1. Subject-level split


def _check_all_folds(sessions: pd.DataFrame, assign: pd.DataFrame) -> None:
    cv.assert_valid_fold_assignment(assign, sessions["subject_id"].unique())
    fold_of = sessions["subject_id"].map(assign.set_index("subject_id")["fold"])
    for f in sorted(assign["fold"].unique()):
        train, test = cv.session_split_subjects(sessions, (fold_of == f).to_numpy())
        assert test and not (train & test)


def test_subject_level_split_on_synthetic_sessions():
    sessions = pd.DataFrame([(f"E_ID{i}", e) for i in range(1, 11) for e in labels.EXERCISES],
                            columns=["subject_id", "exercise"])
    _check_all_folds(sessions, _subject_folds(sessions["subject_id"], k=3))


def test_session_level_split_is_rejected():
    sessions = pd.DataFrame([(f"E_ID{i}", e) for i in range(1, 6) for e in labels.EXERCISES],
                            columns=["subject_id", "exercise"])
    mask = (sessions["exercise"] == 5).to_numpy()  # 운동 기준 분할 → 같은 Subject 가 양쪽에
    with pytest.raises(cv.LeakageError):
        cv.session_split_subjects(sessions, mask)


@needs_interim
def test_subject_level_split_on_real_sessions():
    sessions = _read("manifest_sessions.csv")
    _check_all_folds(sessions, _subject_folds(sessions["subject_id"], k=5))


@needs_interim
def test_one_fold_per_subject_across_exercises():
    """Exercise 별 모델이라도 한 Subject 는 모든 운동에서 같은 fold 다."""
    sessions = _read("manifest_sessions.csv")
    assign = _subject_folds(sessions["subject_id"], k=5)
    s = sessions.assign(fold=sessions["subject_id"].map(assign.set_index("subject_id")["fold"]))
    assert (s.groupby("subject_id")["fold"].nunique() == 1).all()


# ------------------------------------------- 2. Session ↔ Subject 관계


@needs_interim
def test_session_is_subject_x_exercise():
    sessions, subjects = _read("manifest_sessions.csv"), _read("subjects.csv")
    assert not sessions.duplicated(["subject_id", "exercise"]).any()
    assert set(sessions["subject_id"]) == set(subjects["subject_id"])
    per_subject = sessions.groupby("subject_id")["exercise"].apply(set)
    assert (per_subject == set(labels.EXERCISES)).all()
    # Session 의 group 은 그 Subject 의 group 과 같다
    g = sessions.merge(subjects[["subject_id", "group"]], on="subject_id", suffixes=("", "_subj"))
    assert (g["group"] == g["group_subj"]).all()


@needs_interim
def test_label_table_rows_belong_to_known_subjects():
    long, subjects = _read("labels_clinical_long.csv"), _read("subjects.csv")
    assert set(long["subject_id"]) <= set(subjects["subject_id"])
    assert not long.duplicated(["subject_id", "exercise"]).any()


# ------------------------------------------------- 3. 점수 범위


@needs_interim
def test_out_of_range_scores_are_always_flagged():
    """범위 밖 값이 있다면 반드시 label_qc_issues 에 기록되어 있어야 한다."""
    long, issues = _read("labels_clinical_long.csv"), _read("label_qc_issues.csv")
    flagged = {(r.subject_id, r.exercise) for r in issues.itertuples() if r.check == "out_of_range"}
    for s, (lo, hi) in labels.RANGES.items():
        bad = long[long[s].notna() & ~long[s].between(lo, hi)]
        assert {(r.subject_id, r.exercise) for r in bad.itertuples()} <= flagged


# --------------------------------------------- 4. PO + CF 관계를 고치지 않음


@needs_interim
@needs_raw
def test_label_table_equals_raw_values():
    """중간 산출물의 TS/PO/CF 는 원본 값 그대로다 (TS 를 PO+CF 로 다시 계산하지 않음)."""
    files, long = _read("manifest_files.csv"), _read("labels_clinical_long.csv")
    canon = files[(files["kind"] == "clinical") & ~files["is_copy_name"].astype(bool)
                  & ~files["misplaced"].astype(bool)].drop_duplicates("label_subject_id")
    wide = long.pivot(index="subject_id", columns="exercise", values=list(labels.SCORES))
    checked = 0
    for r in canon.itertuples():
        if r.label_subject_id not in wide.index:
            continue
        raw_vals = labels.read_clinical(REAL_RAW / r.path)
        for s in labels.SCORES:
            for e in labels.EXERCISES:
                a, b = raw_vals[f"{s}_Ex{e}"], wide.loc[r.label_subject_id, (s, e)]
                assert (math.isnan(a) and math.isnan(b)) or a == b
        checked += 1
    assert checked > 0


@needs_interim
def test_sum_mismatch_is_reported_not_fixed():
    long, issues = _read("labels_clinical_long.csv"), _read("label_qc_issues.csv")
    d = long.dropna(subset=list(labels.SCORES))
    mism = d[(d["TS"] - (d["PO"] + d["CF"])).abs() > labels.SUM_TOL]
    from_issues = {(r.subject_id, r.exercise) for r in issues.itertuples() if r.check == "sum_mismatch"}
    assert {(r.subject_id, r.exercise) for r in mism.itertuples()} == from_issues


# ----------------------------------------------- 5. 원본 불변


@needs_interim
@needs_raw
def test_raw_signal_and_label_files_match_manifest_hash():
    """inventory 이후 원본 신호·라벨 파일 내용이 바뀌지 않았다 (영상은 크기 때문에 제외)."""
    files = _read("manifest_files.csv")
    sel = files[files["kind"].isin(["position", "orientation", "timestamp", "clinical", "suppinfo"])]
    assert len(sel) > 0
    for r in sel.itertuples():
        p = REAL_RAW / r.path
        assert p.is_file(), r.path
        assert qc.sha256(p) == r.sha256, r.path


def test_label_audit_does_not_modify_raw(raw: Path, tmp_path: Path):  # noqa: F811
    qc.run(raw, tmp_path / "interim")
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in raw.rglob("*") if p.is_file()}
    label_audit.run(raw, tmp_path / "interim", tmp_path / "out")
    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in raw.rglob("*") if p.is_file()}
    assert before == after


# --------------------------------------------- label_audit 표 구조


def test_availability_tables_are_nested_and_separate_units(raw: Path, tmp_path: Path):  # noqa: F811
    qc.run(raw, tmp_path / "interim")
    res = label_audit.run(raw, tmp_path / "interim", tmp_path / "out")
    g = res["label_availability_group"].set_index("Group")
    body = g.drop(index="합계")
    assert (body["Subject"] >= body["Clinical"]).all()
    assert (body["Clinical"] >= body["Score"]).all()
    for s in labels.SCORES:
        assert (body["Score"] >= body[s]).all()
    assert (g.loc["합계"] == body.sum()).all()
    ex = res["label_availability_exercise"]
    assert (ex["sessions_in_grid"] >= ex["sessions_all_scores"]).all()
    assert (ex["sessions_all_scores"] >= ex["sessions_signal_and_scores"]).all()
    # Session 표의 격자 수 = Subject 수 (Group 별, 운동마다)
    per_ex = ex.groupby("group", observed=True)["sessions_in_grid"].agg(["min", "max"])
    for grp, row in per_ex.iterrows():
        assert row["min"] == row["max"] == body.loc[grp, "Subject"]


def test_recheck_targets_are_computed(raw: Path, tmp_path: Path):  # noqa: F811
    qc.run(raw, tmp_path / "interim")
    res = label_audit.run(raw, tmp_path / "interim", tmp_path / "out")
    issues = pd.read_csv(tmp_path / "interim" / "label_qc_issues.csv", encoding="utf-8-sig")
    expected = {(r.subject_id, int(r.exercise)) for r in issues.itertuples() if r.check == "sum_mismatch"}
    assert set(res["flags"]["sum_mismatch"]) == expected
    files = res["label_recheck_files"]
    assert set(files["subject_id"]) == {sid for sid, _ in expected} | set(res["flags"]["irregular"])
    assert (files["n_formula_cells"] >= 0).all()


# --------------------------------------------- 분석 수행 판정 (설계 문서 §4.1)


def test_feasibility_counts_follow_definition(raw: Path, tmp_path: Path):  # noqa: F811
    qc.run(raw, tmp_path / "interim")
    res = label_audit.run(raw, tmp_path / "interim", tmp_path / "out")
    f = res["analysis_feasibility"]
    sessions = pd.read_csv(tmp_path / "interim" / "manifest_sessions.csv", encoding="utf-8-sig")
    long = pd.read_csv(tmp_path / "interim" / "labels_clinical_long.csv", encoding="utf-8-sig")
    d = sessions.merge(long, on=["subject_id", "exercise"], how="left")
    ok = d[d["analyzable_signal"].astype(bool) & d[list(labels.SCORES)].notna().all(axis=1)]
    for ex in labels.EXERCISES:
        row = f[(f["exercise"] == ex) & (f["scope"] == "전체")].iloc[0]
        assert row["n_analyzable_subjects"] == ok.loc[ok["exercise"] == ex, "subject_id"].nunique()
    # 하위 집계는 상위를 넘지 않는다
    assert (f["n_analyzable_subjects"] <= f["n_subjects_in_scope"]).all()
    assert (f["n_analyzable_with_age"] <= f["n_analyzable_subjects"]).all()
    assert (f["n_analyzable_excl_irregular"] <= f["n_analyzable_subjects"]).all()


def test_zero_analyzable_is_never_modeled(raw: Path, tmp_path: Path):  # noqa: F811
    """분석 가능 Subject 가 0명인 범위는 항상 '모델링 불가'이고, 기준이 없으므로 그 외는 보류다."""
    qc.run(raw, tmp_path / "interim")
    f = label_audit.run(raw, tmp_path / "interim", tmp_path / "out")["analysis_feasibility"]
    zero = f["n_analyzable_subjects"] == 0
    assert (f.loc[zero, "status"] == label_audit.FEASIBILITY_NONE).all()
    assert (f.loc[~zero, "status"] == label_audit.FEASIBILITY_PENDING).all()
    assert zero.any()  # 합성 fixture 에는 신호+점수가 없는 범위가 있다
