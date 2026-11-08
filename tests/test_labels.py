"""Label QC 구조 테스트 (REQUIREMENTS §14.3). 합성 xlsx만 사용하고 실제 데이터 수치는 쓰지 않는다."""

import math

import openpyxl
import pandas as pd
import pytest

from kimore.io import ParseError
from kimore.labels import (
    EXERCISES,
    SCORES,
    classify_value,
    clinical_long,
    label_qc,
    read_clinical,
    read_suppinfo,
    score_columns,
    subject_value_format,
)


def _clinical_headers():
    return ["Subject ID"] + [f"clinical {s} Ex#{e}" for s in SCORES for e in EXERCISES]


def _write_xlsx(path, headers, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(headers)
    for r in rows:
        ws.append(r)
    wb.save(path)
    return path


def _valid_scores():
    # TS = PO + CF, 모두 범위 안
    po = [10, 11, 12, 13, 14]
    cf = [20, 21, 22, 23, 24]
    ts = [p + c for p, c in zip(po, cf)]
    return ts + po + cf


def _wide(sid="X_ID1", **overrides):
    vals = dict(zip(score_columns(), _valid_scores()))
    vals.update(overrides)
    return pd.DataFrame([vals], index=pd.Index([sid], name="subject_id"))


def test_score_columns_order():
    cols = score_columns()
    assert len(cols) == len(SCORES) * len(EXERCISES)
    assert cols[0] == "TS_Ex1" and cols[len(EXERCISES)] == "PO_Ex1"
    assert len(set(cols)) == len(cols)


def test_read_clinical_roundtrip(tmp_path):
    p = _write_xlsx(tmp_path / "c.xlsx", _clinical_headers(), [["X_ID1", *_valid_scores()]])
    out = read_clinical(p)
    assert out["subject_in_file"] == "X_ID1"
    assert [out[c] for c in score_columns()] == pytest.approx(_valid_scores())


def test_read_clinical_header_tolerant(tmp_path):
    headers = ["subject id"] + [f"Clinical  {s} Ex {e}" for s in SCORES for e in EXERCISES]
    p = _write_xlsx(tmp_path / "c.xlsx", headers, [["X_ID1", *_valid_scores()]])
    assert read_clinical(p)["TS_Ex1"] == pytest.approx(_valid_scores()[0])


def test_read_clinical_missing_header_raises(tmp_path):
    headers = _clinical_headers()[:-1]
    p = _write_xlsx(tmp_path / "c.xlsx", headers, [["X_ID1", *_valid_scores()[:-1]]])
    with pytest.raises(ParseError):
        read_clinical(p)


def test_read_clinical_empty_scores(tmp_path):
    p = _write_xlsx(tmp_path / "c.xlsx", _clinical_headers(), [["X_ID1"]])
    out = read_clinical(p)
    assert out["subject_in_file"] == "X_ID1"
    assert all(math.isnan(out[c]) for c in score_columns())


def test_read_clinical_no_data_row(tmp_path):
    p = _write_xlsx(tmp_path / "c.xlsx", _clinical_headers(), [])
    out = read_clinical(p)
    assert out["subject_in_file"] is None
    assert all(math.isnan(out[c]) for c in score_columns())


def test_read_suppinfo(tmp_path):
    p = _write_xlsx(tmp_path / "s.xlsx", ["Subject ID", "Group", "Age", "Gender"], [["X_ID1", "E", 40, "F"]])
    out = read_suppinfo(p)
    assert out == {"subject_in_file": "X_ID1", "group": "E", "age": 40.0, "gender": "F"}


def test_read_suppinfo_missing_age(tmp_path):
    p = _write_xlsx(tmp_path / "s.xlsx", ["Subject ID", "Group", "Age", "Gender"], [["X_ID1", "E", None, "M"]])
    assert math.isnan(read_suppinfo(p)["age"])


def test_read_suppinfo_missing_header_raises(tmp_path):
    p = _write_xlsx(tmp_path / "s.xlsx", ["Subject ID", "Group", "Gender"], [["X_ID1", "E", "F"]])
    with pytest.raises(ParseError):
        read_suppinfo(p)


@pytest.mark.parametrize(
    "value,expected",
    [(48, "integer"), (48.0, "integer"), (48.333333333333336, "third"), (13.666666666666666, "third"),
     (28.210081123309372, "irregular"), (None, "missing"), (float("nan"), "missing")],
)
def test_classify_value(value, expected):
    assert classify_value(value) == expected


def test_subject_value_format_priority():
    assert subject_value_format([1, 2.0, 1 / 3]) == "third"
    assert subject_value_format([1, 1 / 3, 0.4256]) == "irregular"
    assert subject_value_format([1, 2, float("nan")]) == "integer"
    assert subject_value_format([float("nan"), None]) == "missing"


def test_label_qc_clean():
    out = label_qc(_wide())
    assert out.empty
    assert list(out.columns) == ["subject_id", "exercise", "check", "detail"]


def test_label_qc_detects_out_of_range():
    out = label_qc(_wide(PO_Ex2=16, CF_Ex2=36, TS_Ex2=52))
    hits = out[out["check"] == "out_of_range"]
    assert set(hits["exercise"]) == {2}
    assert len(hits) == 3


def test_label_qc_detects_negative():
    out = label_qc(_wide(TS_Ex1=-1))
    assert ((out["check"] == "out_of_range") & (out["exercise"] == 1)).any()


def test_label_qc_detects_missing():
    out = label_qc(_wide(PO_Ex4=float("nan")))
    miss = out[out["check"] == "missing"]
    assert list(miss["exercise"]) == [4]
    # 결측이 있으면 합계 검사는 하지 않는다
    assert not ((out["check"] == "sum_mismatch") & (out["exercise"] == 4)).any()


def test_label_qc_detects_sum_mismatch_and_tolerance():
    vals = dict(zip(score_columns(), _valid_scores()))
    out = label_qc(_wide(TS_Ex5=vals["TS_Ex5"] + 0.67))
    mm = out[out["check"] == "sum_mismatch"]
    assert list(mm["exercise"]) == [5]

    within_tol = label_qc(_wide(TS_Ex5=vals["TS_Ex5"] + 0.001))
    assert within_tol.empty


def test_label_qc_exercise_is_int():
    out = label_qc(_wide(PO_Ex3=float("nan")))
    assert pd.api.types.is_integer_dtype(out["exercise"])


def test_clinical_long_shape():
    wide = pd.concat([_wide("A_ID1"), _wide("A_ID2")])
    long = clinical_long(wide)
    assert list(long.columns) == ["subject_id", "exercise", *SCORES]
    assert len(long) == len(wide) * len(EXERCISES)
    assert not long.duplicated(["subject_id", "exercise"]).any()
