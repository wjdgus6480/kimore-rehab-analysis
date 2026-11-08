"""Leakage / CV 규칙 검증 테스트 (REQUIREMENTS §14.4). 합성 데이터만 사용."""

import numpy as np
import pandas as pd
import pytest

from kimore.cv import (
    LeakageError,
    assert_no_subject_overlap,
    assert_valid_fold_assignment,
    forbidden_feature_columns,
    session_split_subjects,
    validate_fold_assignment,
)

SUBJECTS = ["E_ID1", "E_ID2", "NE_ID1", "P_ID1", "S_ID1", "B_ID1"]


def _assignment(subjects, n_folds=3):
    return pd.DataFrame({"subject_id": subjects, "fold": [i % n_folds for i in range(len(subjects))]})


# --- train/test 겹침 ---------------------------------------------------------

def test_disjoint_subjects_pass():
    half = len(SUBJECTS) // 2
    assert_no_subject_overlap(SUBJECTS[:half], SUBJECTS[half:])


def test_overlap_detected_and_named():
    shared = SUBJECTS[0]
    with pytest.raises(LeakageError, match=shared):
        assert_no_subject_overlap(SUBJECTS[:2], [shared, SUBJECTS[-1]])


# --- fold assignment ---------------------------------------------------------

def test_valid_assignment_has_no_problems():
    assign = _assignment(SUBJECTS)
    assert validate_fold_assignment(assign, expected_subjects=SUBJECTS) == []
    assert_valid_fold_assignment(assign, expected_subjects=SUBJECTS)


def test_duplicate_subject_row_detected():
    assign = _assignment(SUBJECTS)
    dup = pd.concat([assign, assign.iloc[[0]]], ignore_index=True)
    problems = validate_fold_assignment(dup)
    assert any("중복" in p for p in problems)


def test_subject_in_multiple_folds_detected():
    assign = _assignment(SUBJECTS)
    first = assign.iloc[[0]].copy()
    first["fold"] = first["fold"] + 1
    problems = validate_fold_assignment(pd.concat([assign, first], ignore_index=True))
    assert any("여러 fold" in p for p in problems)
    with pytest.raises(LeakageError):
        assert_valid_fold_assignment(pd.concat([assign, first], ignore_index=True))


def test_missing_fold_value_detected():
    assign = _assignment(SUBJECTS).astype({"fold": "float"})
    assign.loc[0, "fold"] = np.nan
    problems = validate_fold_assignment(assign)
    assert any("fold 결측" in p for p in problems)


def test_missing_and_extra_subjects_detected():
    assign = _assignment(SUBJECTS[1:] + ["X_ID99"])
    problems = validate_fold_assignment(assign, expected_subjects=SUBJECTS)
    assert any("누락" in p and SUBJECTS[0] in p for p in problems)
    assert any("예상 목록에 없는" in p and "X_ID99" in p for p in problems)


def test_missing_required_column_reported():
    problems = validate_fold_assignment(pd.DataFrame({"subject_id": SUBJECTS}))
    assert problems and "필수 열" in problems[0]


# --- Session 단위 분할 -------------------------------------------------------

def _sessions():
    rows = [(s, ex) for s in SUBJECTS for ex in range(1, 6)]
    return pd.DataFrame(rows, columns=["subject_id", "exercise"])


def test_subject_level_session_split_passes():
    sessions = _sessions()
    test_subjects = set(SUBJECTS[: len(SUBJECTS) // 2])
    mask = sessions["subject_id"].isin(test_subjects)
    train, test = session_split_subjects(sessions, mask)
    assert test == test_subjects
    assert train.isdisjoint(test)
    assert train | test == set(SUBJECTS)


def test_same_subject_other_exercise_split_detected():
    sessions = _sessions()
    target = SUBJECTS[0]
    # 같은 Subject의 운동 하나만 test로 보내는 잘못된 분할
    mask = (sessions["subject_id"] == target) & (sessions["exercise"] == sessions["exercise"].min())
    with pytest.raises(LeakageError, match=target):
        session_split_subjects(sessions, mask)


def test_random_session_split_detected():
    sessions = _sessions()
    mask = np.zeros(len(sessions), dtype=bool)
    mask[::2] = True  # 세션 단위 교대 분할 → 같은 Subject가 양쪽에 들어감
    with pytest.raises(LeakageError):
        session_split_subjects(sessions, mask)


def test_mask_length_mismatch_rejected():
    sessions = _sessions()
    with pytest.raises(LeakageError):
        session_split_subjects(sessions, np.ones(len(sessions) - 1, dtype=bool))


# --- 금지 feature 열 ---------------------------------------------------------

@pytest.mark.parametrize(
    "col",
    [
        "subject_id",
        "group",
        "group_top",
        "subject_folder",
        "rel_path",
        "n_frames",
        "n_rows_raw",
        "duration_s",
        "inferred_ratio",
        "untracked_ratio",
        "median_dt_timestamp",
        "recording_key",
        "recording_date",
        "camera_distance",
        "TS",
        "PO_Ex1",
        "CF_Ex5",
    ],
)
def test_forbidden_columns_detected(col):
    assert forbidden_feature_columns([col]) == [col]


def test_normal_features_pass():
    features = ["shoulder_rom_median", "trunk_tilt_peak", "rep_time_cv", "lr_symmetry_index", "age"]
    assert forbidden_feature_columns(features) == []


def test_mixed_columns_return_only_forbidden():
    cols = ["shoulder_rom_median", "group", "PO_Ex1", "rep_time_cv"]
    assert forbidden_feature_columns(cols) == ["group", "PO_Ex1"]
