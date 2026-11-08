"""파싱·schema 구조 테스트 (REQUIREMENTS §14.2). 합성 fixture만 사용한다."""

from __future__ import annotations

import numpy as np
import pytest

from kimore.io import (
    JOINTS,
    N_JOINTS,
    NO_ORIENTATION_JOINTS,
    ORIENTATION_FIELDS,
    POSITION_FIELDS,
    ParseError,
    frames_aligned,
    is_copy_name,
    orientation_columns,
    parse_label_filename,
    position_columns,
    read_orientation,
    read_position,
    read_timestamp,
    recording_key,
)

N_POS_COLS = 25 * 4
N_ORI_COLS = 25 * 4


def _row(n_cols: int, seed: float = 0.0) -> list[str]:
    return [f"{seed + i * 0.001:.6f}" for i in range(n_cols)]


def _write(tmp_path, name: str, rows: list[list[str]], *, blank_before: int = 0,
           trailing_comma: bool | list[bool] = True) -> object:
    if isinstance(trailing_comma, bool):
        trailing_comma = [trailing_comma] * len(rows)
    lines = [""] * blank_before
    for r, tc in zip(rows, trailing_comma):
        lines.append(",".join(r) + ("," if tc else ""))
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    return p


# ---------- schema ----------

def test_joint_schema():
    assert len(JOINTS) == 25 == N_JOINTS
    assert len(set(JOINTS)) == len(JOINTS)
    assert set(NO_ORIENTATION_JOINTS) <= set(JOINTS)


def test_column_counts_and_order():
    pos = position_columns()
    ori = orientation_columns()
    assert len(pos) == N_POS_COLS
    assert len(ori) == N_ORI_COLS
    assert len(set(pos)) == len(pos)
    assert len(set(ori)) == len(ori)
    for i, joint in enumerate(JOINTS):
        assert pos[4 * i] == f"{joint}_x"
        for k, f in enumerate(POSITION_FIELDS):
            assert pos[4 * i + k] == f"{joint}_{f}"
        for k, f in enumerate(ORIENTATION_FIELDS):
            assert ori[4 * i + k] == f"{joint}_{f}"


# ---------- 정상 파싱 ----------

def test_position_with_leading_blanks_and_trailing_comma(tmp_path):
    rows = [_row(N_POS_COLS, s) for s in range(3)]
    p = _write(tmp_path, "JointPosition_a.csv", rows, blank_before=5)
    df, info = read_position(p)
    assert list(df.columns) == position_columns()
    assert df.shape == (3, N_POS_COLS)
    assert info.n_blank == 5
    assert info.n_frames == 3
    assert info.trailing_comma == "all"
    assert df.iloc[1, 0] == pytest.approx(1.0)


def test_position_without_trailing_comma_same_values(tmp_path):
    rows = [_row(N_POS_COLS, s) for s in range(3)]
    p1 = _write(tmp_path, "a.csv", rows, blank_before=2, trailing_comma=True)
    p2 = _write(tmp_path, "b.csv", rows, blank_before=2, trailing_comma=False)
    df1, i1 = read_position(p1)
    df2, i2 = read_position(p2)
    assert np.array_equal(df1.to_numpy(), df2.to_numpy())
    assert i2.trailing_comma == "none"
    assert (i1.n_frames, i1.n_blank) == (i2.n_frames, i2.n_blank)


def test_mixed_trailing_comma(tmp_path):
    rows = [_row(N_POS_COLS, s) for s in range(2)]
    p = _write(tmp_path, "m.csv", rows, trailing_comma=[True, False])
    _, info = read_position(p)
    assert info.trailing_comma == "mixed"


def test_orientation_parses(tmp_path):
    p = _write(tmp_path, "JointOrientation_a.csv", [_row(N_ORI_COLS)], blank_before=1)
    df, info = read_orientation(p)
    assert list(df.columns) == orientation_columns()
    assert info.n_frames == 1


def test_blank_line_in_middle_tolerated(tmp_path):
    p = tmp_path / "mid.csv"
    r = ",".join(_row(N_POS_COLS)) + ","
    p.write_text(f"\n{r}\n\n{r}\n", encoding="ascii")
    _, info = read_position(p)
    assert info.n_frames == 2


# ---------- 잘못된 형식 탐지 ----------

def test_wrong_column_count_raises(tmp_path):
    rows = [_row(N_POS_COLS), _row(N_POS_COLS - 1)]
    p = _write(tmp_path, "bad.csv", rows)
    with pytest.raises(ParseError):
        read_position(p)


def test_orientation_file_read_as_position_wrong_width(tmp_path):
    p = _write(tmp_path, "ts.csv", [["1"], ["2"]])
    with pytest.raises(ParseError):
        read_position(p)


def test_non_numeric_raises(tmp_path):
    r = _row(N_POS_COLS)
    r[10] = "abc"
    p = _write(tmp_path, "bad.csv", [r])
    with pytest.raises(ParseError):
        read_position(p)


def test_nan_string_raises(tmp_path):
    r = _row(N_POS_COLS)
    r[0] = "nan"
    p = _write(tmp_path, "nan.csv", [r])
    with pytest.raises(ParseError):
        read_position(p)


@pytest.mark.parametrize("content", ["", "\n\n\n", ",,,\n,\n"])
def test_empty_or_blank_only_raises(tmp_path, content):
    p = tmp_path / "empty.csv"
    p.write_text(content, encoding="ascii")
    with pytest.raises(ParseError):
        read_position(p)
    with pytest.raises(ParseError):
        read_timestamp(p)


# ---------- timestamp / alignment ----------

def test_timestamp_integer_dtype(tmp_path):
    p = _write(tmp_path, "TimeStamp_a.csv", [["20622112895"], ["20622442942"]], blank_before=3)
    ts, info = read_timestamp(p)
    assert ts.dtype == np.int64
    assert info.n_frames == 2
    assert ts.iloc[1] > ts.iloc[0]


def test_frames_aligned():
    assert frames_aligned(10, 10)
    assert frames_aligned(10, 10, 10)
    assert not frames_aligned(10, 9)
    assert not frames_aligned(10, 10, 11)


def test_alignment_detects_mismatch_from_files(tmp_path):
    pos = _write(tmp_path, "JointPosition_k.csv", [_row(N_POS_COLS, s) for s in range(3)])
    ts = _write(tmp_path, "TimeStamp_k.csv", [["1"], ["2"]])
    _, pi = read_position(pos)
    _, ti = read_timestamp(ts)
    assert not frames_aligned(pi.n_frames, ti.n_frames)


# ---------- 파일명 규칙 ----------

def test_is_copy_name():
    assert is_copy_name("ClinicalAssessment_E_ID15(1).xlsx")
    assert is_copy_name("feat_extract_Ex4(1).m")
    assert not is_copy_name("ClinicalAssessment_E_ID15.xlsx")
    assert not is_copy_name("JointPosition011214_103748.csv")


def test_parse_label_filename():
    assert parse_label_filename("ClinicalAssessment_E_ID1.xlsx") == ("ClinicalAssessment", "E_ID1", False)
    assert parse_label_filename("SuppInfo_NE_ID24(1).xlsx") == ("SuppInfo", "NE_ID24", True)
    assert parse_label_filename("ClinicalAssessment_P_ID3(1).xlsx") == ("ClinicalAssessment", "P_ID3", True)
    assert parse_label_filename("JointPosition011214_103748.csv") is None
    assert parse_label_filename("SuppInfo_X_ID1.xlsx") is None
    assert parse_label_filename("Clinical_E_ID1.xlsx") is None


def test_recording_key_pairs_files():
    k1 = recording_key("JointPosition011214_103748.csv")
    k2 = recording_key("TimeStamp011214_103748.csv")
    k3 = recording_key("JointOrientation011214_103748.csv")
    assert k1 == k2 == k3 == "011214_103748"
    assert recording_key("JointPosition011214_103748(1).csv") == k1
    assert recording_key("SuppInfo_E_ID1.xlsx") is None
