"""Gate 3.5 비교 도구 구조 테스트 — 합성 트리만 사용한다 (실제 데이터 수치 하드코딩 없음)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kimore import compare_sources as cs
from kimore import labels
from test_qc import _full_scores, _write_clinical, raw  # noqa: F401


def _snapshot(root: Path) -> dict:
    return {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*") if p.is_file()}


@pytest.fixture()
def pair(raw: Path, tmp_path: Path):  # noqa: F811
    official = tmp_path / "official"
    shutil.copytree(raw, official)
    return raw, official


def test_identical_trees_report_no_difference(pair, tmp_path):
    cur, off = pair
    res = cs.run(cur, off, tmp_path / "out")
    f = res["file_comparison"]
    assert set(f["status"]) <= {"identical", "both_not_hashed"}
    assert res["clinical_comparison"]["same_scores"].all()
    assert (res["analysis_feasibility_comparison"]["diff_analyzable"] == 0).all()
    for name in ("file_comparison", "session_comparison", "clinical_comparison", "signal_comparison",
                 "analysis_feasibility_comparison", "inventory_comparison"):
        assert (tmp_path / "out" / f"{name}.csv").is_file()


def test_differences_are_detected_and_reported_as_values(pair, tmp_path):
    cur, off = pair
    # 공식 쪽: 점수 하나 변경, 신호 파일 하나 제거, 파일 하나 추가
    scores = _full_scores()
    scores["PO_Ex2"] = 12.5
    _write_clinical(off / "CG/Expert/E_ID1/Es1/Label/ClinicalAssessment_E_ID1.xlsx", "E_ID1", scores)
    (off / "CG/Expert/E_ID1/Es1/Raw/TimeStamp_a.csv").unlink()
    (off / "CG/Expert/E_ID1/Es4").mkdir(parents=True)
    shutil.copy(cur / "CG/Expert/E_ID1/Es1/Raw/JointPosition_a.csv", off / "CG/Expert/E_ID1/Es4/JointPosition_z.csv")

    res = cs.run(cur, off, tmp_path / "out")
    f = res["file_comparison"].set_index("path")
    assert f.loc["CG/Expert/E_ID1/Es1/Label/ClinicalAssessment_E_ID1.xlsx", "status"] == "content_differs"
    assert f.loc["CG/Expert/E_ID1/Es1/Raw/TimeStamp_a.csv", "status"] == "current_only"
    assert f.loc["CG/Expert/E_ID1/Es4/JointPosition_z.csv", "status"] == "official_only"

    cl = res["clinical_comparison"]
    row = cl[(cl["subject_id"] == "E_ID1") & (cl["exercise"] == 2)].iloc[0]
    assert not row["same_PO"]
    assert row["PO_official"] == 12.5 and row["PO_current"] != 12.5

    s = res["session_comparison"]
    row = s[(s["subject_id"] == "E_ID1") & (s["exercise"] == 1)].iloc[0]
    assert row["has_timestamp_current"] and not row["has_timestamp_official"]


def test_gate3_targets_are_always_marked(pair, tmp_path):
    cur, off = pair
    cl = cs.run(cur, off, tmp_path / "out")["clinical_comparison"]
    d = cl[cl["delta_current"].abs() > labels.SUM_TOL]
    assert d["recheck_target"].all()


def test_both_trees_are_read_only(pair, tmp_path):
    cur, off = pair
    before = (_snapshot(cur), _snapshot(off))
    cs.run(cur, off, tmp_path / "out")
    assert (_snapshot(cur), _snapshot(off)) == before


def test_refuses_output_inside_source_and_same_paths(pair, tmp_path):
    cur, off = pair
    with pytest.raises(ValueError):
        cs.run(cur, off, off / "reports")
    with pytest.raises(ValueError):
        cs.run(cur, cur, tmp_path / "out")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cs.run(cur, off, foreign)
