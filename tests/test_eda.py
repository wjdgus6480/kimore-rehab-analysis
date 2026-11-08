"""EDA 단계 구조 테스트 (REQUIREMENTS §14.1: 결과 수치를 하드코딩하지 않고 불변 조건만 검사).

test_qc 의 합성 원본 트리를 재사용한다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kimore import eda, labels, qc
from test_qc import raw  # noqa: F401  (pytest fixture 재사용)


@pytest.fixture()
def eda_result(raw: Path, tmp_path: Path):  # noqa: F811
    qc.run(raw, tmp_path / "interim")
    return eda.run(raw, tmp_path / "interim", tmp_path / "reports", make_figures=True)


def test_outputs_written(eda_result, tmp_path):
    rep = tmp_path / "reports"
    assert (rep / "EDA_report.md").read_text(encoding="utf-8").startswith("# KIMORE EDA")
    pngs = list((rep / "figures").glob("*.png"))
    assert len(pngs) == len(eda_result["figures"]) > 0
    assert (tmp_path / "interim" / "eda" / "eda_sessions.csv").is_file()
    for f in eda_result["figures"]:
        assert f.question  # 모든 그래프는 질문 하나에 답한다


def test_session_grid_preserved(eda_result):
    es = eda_result["es"]
    sessions = eda_result["interim"]["sessions"]
    assert len(es) == len(sessions)
    assert not es.duplicated(["subject_id", "exercise"]).any()


def test_delta_matches_label_qc(eda_result):
    es = eda_result["es"]
    issues = eda_result["interim"]["issues"]
    from_qc = {(r.subject_id, r.exercise) for r in issues.itertuples() if r.check == "sum_mismatch"}
    from_eda = {(r.subject_id, r.exercise) for r in es[es["delta_nonzero"]].itertuples()}
    assert from_eda == from_qc
    d = es.dropna(subset=list(labels.SCORES))
    assert np.allclose(d["delta"], d["TS"] - (d["PO"] + d["CF"]))


def test_out_of_range_matches_label_qc(eda_result):
    issues = eda_result["interim"]["issues"]
    from_qc = {(r.subject_id, r.exercise) for r in issues.itertuples() if r.check == "out_of_range"}
    t = eda_result["tables"]["out_of_range"]
    assert {(r.subject_id, r.exercise) for r in t.itertuples()} == from_qc


def test_ceiling_is_computed_from_data(eda_result):
    es = eda_result["es"]
    t = eda_result["tables"]["ceiling_by_group"].set_index("group")
    for g, row in t.iterrows():
        v = es.loc[(es["group"] == g) & es["TS"].notna(), "TS"]
        assert row["n"] == len(v)
        assert row["k_ts_ge_45"] == int((v >= eda.CEILING_TS).sum())
        assert row["ci_low"] <= row["ratio"] <= row["ci_high"]


def test_missing_scores_not_imputed(eda_result):
    es = eda_result["es"]
    long = eda_result["interim"]["long"]
    merged = es.merge(long, on=["subject_id", "exercise"], suffixes=("", "_src"))
    for s in labels.SCORES:
        assert (merged[s].isna() == merged[f"{s}_src"].isna()).all()


def test_signal_scan_skips_multiple_recordings(eda_result):
    sk = eda_result["signal"]["skipped"]
    # 합성 트리의 E_ID2/Es1 은 녹화 2개 → 자동 선택하지 않는다
    assert ((sk["subject_id"] == "E_ID2") & (sk["exercise"] == 1)).any()
    ts = eda_result["signal"]["ts_sessions"]
    assert not ((ts["subject_id"] == "E_ID2") & (ts["exercise"] == 1)).any()


def test_raw_is_not_modified(raw: Path, tmp_path: Path):  # noqa: F811
    qc.run(raw, tmp_path / "interim")
    before = {p: p.stat().st_mtime_ns for p in raw.rglob("*") if p.is_file()}
    eda.run(raw, tmp_path / "interim", tmp_path / "reports", make_figures=False)
    after = {p: p.stat().st_mtime_ns for p in raw.rglob("*") if p.is_file()}
    assert before == after


def test_refuses_foreign_output_dir(raw: Path, tmp_path: Path):  # noqa: F811
    qc.run(raw, tmp_path / "interim")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("user file", encoding="utf-8")
    with pytest.raises(FileExistsError):
        eda.run(raw, tmp_path / "interim", foreign, make_figures=False)
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "user file"


# ----------------------------------------------------------- stat helpers


def test_wilson_ci_bounds():
    for k, n in [(0, 5), (5, 5), (3, 10)]:
        lo, hi = eda.wilson_ci(k, n)
        assert 0 <= lo <= k / n <= hi <= 1
    assert all(math.isnan(v) for v in eda.wilson_ci(0, 0))


def test_spearman_and_cliffs_delta():
    x = pd.Series([1, 2, 3, 4, np.nan])
    assert eda.spearman(x, x ** 2) == (pytest.approx(1.0), 4)
    assert eda.spearman(x, -x)[0] == pytest.approx(-1.0)
    assert eda.cliffs_delta([5, 6], [1, 2]) == 1.0
    assert eda.cliffs_delta([1, 2], [1, 2]) == 0.0
