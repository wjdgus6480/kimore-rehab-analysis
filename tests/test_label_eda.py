"""Gate 2 라벨·Age EDA 테스트 (REQUIREMENTS §14). 합성 데이터만 쓰고 실제 데이터 수치를 하드코딩하지 않는다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kimore import label_eda as L
from kimore import label_eda_plots, label_eda_report, labels

COLS = labels.score_columns()


def make_subjects(spec: list[dict]) -> pd.DataFrame:
    """spec: subject_id, group, age(None 가능), scores(dict 운동→(TS,PO,CF)), has_file(기본 True)."""
    rows = []
    for s in spec:
        top = L.qc.GROUPS[s["group"]][0]
        row = {
            "subject_id": s["subject_id"],
            "group": s["group"],
            "group_top": top,
            "cohort": L.COHORT_OF_TOP[top],
            "age": np.nan if s.get("age") is None else float(s["age"]),
            "gender": None if s.get("age") is None else "F",
            "supp_group": None if s.get("age") is None else s["group"][0],
            "has_suppinfo": s.get("age") is not None,
            "has_age": s.get("age") is not None,
            "C1_has_clinical_file": s.get("has_file", True),
        }
        for c in COLS:
            row[c] = np.nan
        for ex, (ts, po, cf) in s.get("scores", {}).items():
            row[f"TS_Ex{ex}"], row[f"PO_Ex{ex}"], row[f"CF_Ex{ex}"] = ts, po, cf
        vals = [row[c] for c in COLS]
        row["n_scores_present"] = int(pd.notna(vals).sum())
        row["C2_has_any_score"] = row["n_scores_present"] > 0
        row["C3_has_all_scores"] = row["n_scores_present"] == len(COLS)
        row["value_format"] = labels.subject_value_format(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def full(ts, po, cf):
    return {ex: (ts, po, cf) for ex in labels.EXERCISES}


@pytest.fixture()
def subjects() -> pd.DataFrame:
    spec = [
        {"subject_id": "E_ID1", "group": "Expert", "age": 25, "scores": full(48, 14, 34)},
        {"subject_id": "E_ID2", "group": "Expert", "age": 30, "scores": full(45, 13, 32)},
        {"subject_id": "NE_ID1", "group": "NotExpert", "age": 28, "scores": full(44, 13, 31)},
        {"subject_id": "NE_ID2", "group": "NotExpert", "age": None, "scores": full(40, 12, 28)},
        {"subject_id": "B_ID1", "group": "BackPain", "age": 55, "scores": full(35, 10, 25)},
        {"subject_id": "B_ID2", "group": "BackPain", "age": 60,
         "scores": {1: (30.4, 10.1234, 20.2766), 2: (31, 10, 21)}},  # 불규칙 소수, 일부 운동만
        {"subject_id": "P_ID1", "group": "Parkinson", "age": 70, "scores": full(25, 8, 17)},
        {"subject_id": "S_ID1", "group": "Stroke", "age": 65, "scores": full(30, 9, 21)},
        {"subject_id": "S_ID2", "group": "Stroke", "age": 50, "has_file": False},
    ]
    return make_subjects(spec)


@pytest.fixture()
def sessions(subjects) -> pd.DataFrame:
    return L.to_sessions(subjects)


# ------------------------------------------------------------ 1. score summary


def test_summarize_scores_values():
    df = pd.DataFrame({"group": ["Expert"] * 4, "TS": [10.0, 20.0, 30.0, 40.0], "PO": [1.0] * 4, "CF": [9.0] * 4})
    out = L.summarize_scores(df, ["group"]).set_index("score")
    ts = out.loc["TS"]
    assert ts["n"] == 4
    assert ts["mean"] == pytest.approx(25.0)
    assert ts["median"] == pytest.approx(25.0)
    assert ts["min"] == 10 and ts["max"] == 40
    assert ts["iqr"] == pytest.approx(ts["q3"] - ts["q1"])
    assert set(out["unit"]) == {"Session"}


# -------------------------------------------------------------- 2. missing


def test_missing_values_are_excluded_not_filled():
    df = pd.DataFrame({"TS": [10.0, np.nan, 30.0], "PO": [np.nan] * 3, "CF": [5.0, 6.0, np.nan]})
    out = L.summarize_scores(df).set_index("score")
    assert out.loc["TS", "n"] == 2 and out.loc["TS", "mean"] == pytest.approx(20.0)
    assert out.loc["PO", "n"] == 0 and np.isnan(out.loc["PO", "mean"])
    assert out.loc["CF", "n"] == 2


def test_partial_exercise_missing_kept_as_nan(sessions):
    b2 = sessions[sessions.subject_id == "B_ID2"]
    assert len(b2) == len(labels.EXERCISES)
    assert b2["TS"].notna().sum() == 2


# ---------------------------------------------------------------- 3. ranges


def test_range_violations_detected():
    df = pd.DataFrame({"subject_id": ["X", "Y", "Z"], "exercise": [1, 1, 1],
                       "TS": [51.0, 40.0, np.nan], "PO": [10.0, -1.0, 5.0], "CF": [30.0, 20.0, 36.0]})
    bad = L.check_score_ranges(df)
    assert set(zip(bad.subject_id, bad.score)) == {("X", "TS"), ("Y", "PO"), ("Z", "CF")}
    obs = L.observed_ranges(df).set_index("score")
    assert obs.loc["TS", "n_missing"] == 1
    assert obs.loc["TS", "n_out_of_range"] == 1


def test_no_violations_in_valid_data(sessions):
    assert L.check_score_ranges(sessions).empty


# ------------------------------------------------------- 4. TS - (PO + CF)


def test_sum_residual():
    df = pd.DataFrame({"subject_id": ["A", "B", "C", "D"], "group": ["Expert"] * 4, "exercise": [1] * 4,
                       "TS": [45.0, 44.0, 40.0, np.nan], "PO": [13.0, 13.0, 10.0, 10.0],
                       "CF": [32.0, 30.333, 30.004, 20.0]})
    r = L.sum_residual(df).set_index("subject_id")
    assert "D" not in r.index  # 결측은 계산하지 않음
    assert r.loc["A", "ts_minus_sum"] == pytest.approx(0) and not r.loc["A", "nonzero"]
    assert r.loc["B", "nonzero"]
    assert not r.loc["C", "nonzero"]  # 허용 오차 이내


# ----------------------------------------------------------- 5. Age missing


def test_suppinfo_completeness_separates_counts(subjects):
    s = L.suppinfo_completeness(subjects)
    assert s["n_subjects"] == len(subjects)
    assert s["n_age"] == int(subjects.has_age.sum())
    assert s["n_age"] < s["n_subjects"]
    assert s["age_missing_subjects"] == ["NE_ID2"]
    assert s["n_age_and_C2"] == int((subjects.has_age & subjects.C2_has_any_score).sum())
    total = s["table"].set_index("group").loc["합계"]
    assert total["subjects"] == len(subjects)


def test_age_analyses_use_only_age_available(subjects, sessions):
    ov = L.age_overlap(subjects)
    assert ov["n_healthy"] + ov["n_patient"] == int(subjects.has_age.sum())
    ac = L.age_score_correlation(sessions)
    row = ac[(ac.cohort == "전체") & (ac.exercise == "평균") & (ac.score == "TS")].iloc[0]
    subj = L.subject_scores(sessions)
    assert row["n"] == int(subj["age"].notna().sum())


def test_age_overlap_classes():
    def subj(h, p):
        spec = [{"subject_id": f"E_ID{i}", "group": "Expert", "age": a} for i, a in enumerate(h)]
        spec += [{"subject_id": f"P_ID{i}", "group": "Parkinson", "age": a} for i, a in enumerate(p)]
        return make_subjects(spec)

    same = list(range(30, 70, 2))
    assert L.age_overlap(subj(same, same))["class"] == "A"
    assert L.age_overlap(subj([20, 22, 25], [70, 75, 80]))["class"] == "C"
    res = L.age_overlap(subj(same, same))
    assert res["n_matched_pairs"] == len(same)


# --------------------------------------------------- 6. correlation format


def test_correlation_format():
    c = L.correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 11])
    assert set(c) == {"n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"}
    assert c["n"] == 5 and c["spearman_rho"] == pytest.approx(1.0)
    small = L.correlation([1, 2], [3, 4])
    assert small["n"] == 2 and np.isnan(small["spearman_rho"])
    with_nan = L.correlation([1, np.nan, 3, 4], [1, 2, np.nan, 4])
    assert with_nan["n"] == 2


def test_age_score_correlation_columns(sessions):
    ac = L.age_score_correlation(sessions)
    assert {"cohort", "unit", "exercise", "score", "n", "rho", "p"} <= set(ac.columns)
    assert set(ac.cohort) == {"전체", *L.COHORT_ORDER}


# ----------------------------------------------------------- 7. sensitivity


def test_irregular_subjects_detected(subjects):
    assert L.irregular_subjects(subjects) == ["B_ID2"]


def test_label_sensitivity_excludes_only_flagged(subjects, sessions):
    flagged = L.irregular_subjects(subjects)
    res = L.compare_label_sensitivity(subjects, sessions, flagged)
    n_a, n_b = res["n_subjects"]
    assert n_a - n_b == len(flagged)
    assert res["n_sessions"][0] - res["n_sessions"][1] == len(flagged) * len(labels.EXERCISES)
    g = res["by_group"].set_index(["group", "score"])
    assert g.loc[("Expert", "TS"), "mean_diff_B_minus_A"] == pytest.approx(0)  # 영향 없는 그룹은 그대로
    assert g.loc[("BackPain", "TS"), "n_B"] < g.loc[("BackPain", "TS"), "n_A"]
    assert {"relation", "spearman_rho_A", "spearman_rho_B", "rho_diff_B_minus_A"} <= set(res["correlations"].columns)


def test_sensitivity_with_no_flagged_is_identical(subjects, sessions):
    res = L.compare_label_sensitivity(subjects, sessions, [])
    assert res["n_subjects"][0] == res["n_subjects"][1]
    assert (res["by_group"]["mean_diff_B_minus_A"].fillna(0) == 0).all()


# -------------------------------------------------- 8. Subject vs Session unit


def test_session_table_is_subject_x_exercise(subjects, sessions):
    n_c1 = int(subjects.C1_has_clinical_file.sum())
    assert len(sessions) == n_c1 * len(labels.EXERCISES)
    assert not sessions.duplicated(["subject_id", "exercise"]).any()
    assert "S_ID2" not in set(sessions.subject_id)  # 점수 파일 없는 대상자는 Session 표에 없음


def test_subject_scores_one_row_per_subject(sessions):
    subj = L.subject_scores(sessions)
    assert subj.subject_id.is_unique
    assert len(subj) == sessions.groupby("subject_id")["TS"].count().gt(0).sum()
    b2 = subj.set_index("subject_id").loc["B_ID2"]
    assert b2["n_exercises_scored"] == 2


def test_group_tests_use_subject_counts(subjects, sessions):
    t = L.group_tests(subjects, sessions)
    assert set(t["unit"]) == {"Subject"}
    kw = t[(t.test == "Kruskal-Wallis") & (t.value == "TS")].iloc[0]
    assert kw["n"] == L.subject_scores(sessions)["TS"].notna().sum()
    assert kw["n"] < len(sessions)
    assert kw["unstable"]  # 합성 데이터는 그룹이 작다


def test_ceiling_table():
    df = pd.DataFrame({"group": ["Expert"] * 4, "TS": [50.0, 45.0, 44.9, np.nan]})
    c = L.ceiling_table(df, "TS").iloc[0]
    assert c["threshold"] == pytest.approx(45.0)
    assert c["n"] == 3 and c["n_near_ceiling"] == 2 and c["n_at_max"] == 1


# --------------------------------------------------------- end-to-end smoke


def test_report_and_figures_smoke(subjects, sessions, tmp_path):
    res = L.analyze(subjects, sessions)
    figs = label_eda_plots.make_figures(res, tmp_path / "fig")
    assert all(p.is_file() and p.stat().st_size > 0 for p in figs)
    report = label_eda_report.write_report(res, figs, tmp_path / "docs" / "r.md", generated="test")
    text = report.read_text(encoding="utf-8")
    for heading in ("## 1. 분석 목적", "## 5. Label Sensitivity Analysis", "## 9. 주 분석 후보", "## 10. Gate 2"):
        assert heading in text
    assert "](../fig/" in text  # 그림은 상대경로로 연결
    for banned in ("진단한다", "오류다", "효과를 증명"):
        assert banned not in text


def test_analysis_module_has_no_plotting_dependency():
    """분석 모듈은 그림 없이 쓸 수 있어야 한다 (분석/시각화 책임 분리)."""
    import inspect

    assert "matplotlib" not in inspect.getsource(L)


def test_call_unit_functions_return_data(subjects, sessions):
    age = L.analyze_age_confounding(subjects, sessions)
    assert {"completeness", "age_by_group", "age_by_cohort", "age_score_spearman", "overlap"} <= set(age)
    assert isinstance(age["age_score_spearman"], pd.DataFrame)
    assert age["overlap"]["class"] in {"A", "B", "C"}
    sens = L.analyze_label_sensitivity(subjects, sessions)
    assert sens["flagged"] == L.irregular_subjects(subjects)
    res = L.analyze(subjects, sessions)
    assert res["overlap"]["class"] == age["overlap"]["class"]
