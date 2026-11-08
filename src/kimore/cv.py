"""Subject 단위 분할 규칙 검증 (REQUIREMENTS §8, §9, §14.4).

CV 방식(StratifiedGroupKFold, GroupKFold, LOSO 등)은 Gate 3에서 결정한다.
이 모듈은 분할기를 만들지 않고, 어떤 분할이든 지켜야 할 규칙만 검사한다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


class LeakageError(ValueError):
    """Subject leakage 또는 분할 규칙 위반."""


# 식별·메타데이터 열: feature로 사용 금지
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "subject_id",
        "subject",
        "group",
        "group_top",
        "subject_folder",
        "file",
        "filename",
        "file_name",
        "path",
        "rel_path",
        "sha256",
        "file_hash",
    }
)

# 그룹을 대신 설명할 수 있는 대리 변수 패턴 (QC 전용)
PROXY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"n_frames",
        r"n_rows",
        r"duration",
        r"inferred_ratio",
        r"untracked_ratio",
        r"timestamp",
        r"recording",
        r"date",
        r"camera_distance",
        r"(^|_)path($|_)",
        r"(^|_)file($|_)",
    )
)

# 임상 점수 계열 (target 또는 target과 직접 관련된 열)
SCORE_PATTERN: re.Pattern[str] = re.compile(r"^(TS|PO|CF)(_|$)")


def _as_set(values: Iterable) -> set:
    return {v for v in values if not (isinstance(v, float) and np.isnan(v))}


def assert_no_subject_overlap(train_subjects: Iterable, test_subjects: Iterable) -> None:
    """train과 test의 Subject가 겹치면 LeakageError."""
    overlap = _as_set(train_subjects) & _as_set(test_subjects)
    if overlap:
        raise LeakageError(f"train/test에 같은 Subject가 있습니다: {sorted(map(str, overlap))}")


def validate_fold_assignment(
    assign: pd.DataFrame, expected_subjects: Iterable | None = None
) -> list[str]:
    """fold assignment(subject_id, fold)의 문제 목록. 문제가 없으면 []."""
    problems: list[str] = []
    missing_cols = {"subject_id", "fold"} - set(assign.columns)
    if missing_cols:
        return [f"필수 열이 없습니다: {sorted(missing_cols)}"]

    sid = assign["subject_id"]
    if sid.isna().any():
        problems.append(f"subject_id 결측 행 {int(sid.isna().sum())}개")

    na_fold = assign.loc[assign["fold"].isna(), "subject_id"]
    if len(na_fold):
        problems.append(f"fold 결측 Subject: {sorted(map(str, na_fold.dropna().unique()))}")

    dup = sid[sid.duplicated(keep=False) & sid.notna()]
    if len(dup):
        problems.append(f"subject_id 중복 행: {sorted(map(str, dup.unique()))}")

    folds_per_subject = assign.dropna(subset=["subject_id", "fold"]).groupby("subject_id")["fold"].nunique()
    multi = folds_per_subject[folds_per_subject > 1]
    if len(multi):
        problems.append(f"여러 fold에 들어간 Subject: {sorted(map(str, multi.index))}")

    if expected_subjects is not None:
        expected = _as_set(expected_subjects)
        present = _as_set(sid)
        absent = expected - present
        extra = present - expected
        if absent:
            problems.append(f"fold assignment에 누락된 Subject: {sorted(map(str, absent))}")
        if extra:
            problems.append(f"예상 목록에 없는 Subject: {sorted(map(str, extra))}")

    return problems


def assert_valid_fold_assignment(
    assign: pd.DataFrame, expected_subjects: Iterable | None = None
) -> None:
    """validate_fold_assignment의 문제가 하나라도 있으면 LeakageError."""
    problems = validate_fold_assignment(assign, expected_subjects)
    if problems:
        raise LeakageError("; ".join(problems))


def session_split_subjects(
    sessions: pd.DataFrame, test_mask: Sequence[bool] | pd.Series | np.ndarray
) -> tuple[set, set]:
    """Session 단위 분할의 (train, test) Subject 집합. 같은 Subject가 양쪽에 있으면 LeakageError."""
    if "subject_id" not in sessions.columns:
        raise LeakageError("sessions에 subject_id 열이 없습니다")
    mask = np.asarray(test_mask, dtype=bool)
    if mask.shape[0] != len(sessions):
        raise LeakageError("test_mask 길이가 sessions 행 수와 다릅니다")
    test_subjects = _as_set(sessions.loc[mask, "subject_id"])
    train_subjects = _as_set(sessions.loc[~mask, "subject_id"])
    assert_no_subject_overlap(train_subjects, test_subjects)
    return train_subjects, test_subjects


def forbidden_feature_columns(columns: Iterable[str]) -> list[str]:
    """feature로 쓰면 안 되는 열(식별자·메타데이터·대리 변수·점수 계열) 목록."""
    bad: list[str] = []
    for col in columns:
        name = str(col)
        if (
            name.lower() in FORBIDDEN_FEATURE_COLUMNS
            or SCORE_PATTERN.match(name)
            or any(p.search(name) for p in PROXY_PATTERNS)
        ):
            bad.append(name)
    return bad
