"""Gate 2 — 라벨·SuppInfo EDA 와 Age confounding 예비 분석 (REQUIREMENTS §4.5~§4.8, §10).

사용:
    python -m kimore.label_eda --raw data/raw --report docs/label_eda_report.md --fig-dir docs/figures/label_eda

분석 단위 (혼동 금지)
- Subject 표: 대상자 1명 = 1행. Age, SuppInfo, 운동 평균 점수.
- Session 표: 대상자 1명 × 운동 1개 = 1행. 점수(TS/PO/CF). ClinicalAssessment 파일이 있는 대상자만.
  같은 대상자의 Session 5개는 독립이 아니므로 대상자 간 검정은 Subject 표로 한다.

원칙
- 원본은 kimore.qc 를 통해 읽기만 한다. 값을 보정·추정하지 않는다.
- 모든 수치는 실행 시 데이터에서 계산한다.
- 이 단계는 EDA 다. 모델, 분할, CV, feature engineering 은 하지 않는다.
- 불규칙 소수 라벨은 '출처 확인이 필요한 라벨'로만 다루며 오류로 간주하지 않는다.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from kimore import labels, qc

SCORES: tuple[str, ...] = labels.SCORES
GROUP_ORDER: tuple[str, ...] = tuple(qc.GROUPS)
COHORT_OF_TOP = {"CG": "건강군", "GPP": "환자군"}  # 폴더 정의 그대로 (REQUIREMENTS §3.1)
COHORT_ORDER: tuple[str, ...] = ("건강군", "환자군")
NEAR_CEILING_FRAC = 0.9  # 만점의 90% 이상 = '상한 근처' (TS 50 → 45)
MIN_N_STABLE = 5  # 그룹 표본이 이보다 작으면 검정 해석 불안정으로 표시
AGE_BIN_WIDTH = 10
MATCH_CALIPER = 5.0  # 연령 매칭 허용 차이(세)
OVERLAP_MIN_PER_BIN = 3

# 연령 overlap 판정 규칙 (분석 전에 정한 휴리스틱; 보고서에 그대로 기록)
OVERLAP_RULE = (
    "A: 두 군 모두 공통 연령 구간 안에 75% 이상, 그리고 양쪽 모두 3명 이상인 10세 구간이 3개 이상 / "
    "C: 공통 구간이 없거나 어느 한 군이라도 공통 구간 안 비율이 25% 미만 / B: 그 외"
)


# ---------------------------------------------------------------------- loading


def load_label_tables(raw_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본에서 (Subject 표, Session 표)를 만든다. kimore.qc 의 로딩을 재사용한다."""
    raw_root = Path(raw_root)
    files = qc.scan_files(raw_root, compute_hash=False)
    subjects, _, _ = qc.build_subjects(raw_root, files)
    subjects = subjects.copy()
    subjects["cohort"] = subjects["group_top"].map(COHORT_OF_TOP)
    return subjects, to_sessions(subjects)


def to_sessions(subjects: pd.DataFrame) -> pd.DataFrame:
    """Subject 표(wide 점수 열 포함) → Session 표. ClinicalAssessment 파일이 있는 대상자만."""
    cols = labels.score_columns()
    has_file = subjects.get("C1_has_clinical_file", pd.Series(True, index=subjects.index)).fillna(False)
    sub = subjects[has_file.astype(bool)]
    wide = sub.set_index("subject_id").reindex(columns=cols)
    long = labels.clinical_long(wide)
    meta = sub[["subject_id", "group", "group_top", "cohort", "age", "value_format"]]
    out = long.merge(meta, on="subject_id", how="left", validate="many_to_one")
    return out.sort_values(["subject_id", "exercise"]).reset_index(drop=True)


def subject_scores(sessions: pd.DataFrame) -> pd.DataFrame:
    """Session 표 → Subject 표 (운동별 점수의 대상자 평균). 결측 운동은 평균에서 제외하고 개수를 남긴다."""
    agg = sessions.groupby("subject_id").agg(
        **{s: (s, "mean") for s in SCORES},
        n_exercises_scored=("TS", "count"),
    )
    meta = sessions.drop_duplicates("subject_id").set_index("subject_id")[
        ["group", "group_top", "cohort", "age", "value_format"]
    ]
    out = meta.join(agg).reset_index()
    return out[out["n_exercises_scored"] > 0].reset_index(drop=True)


def irregular_subjects(subjects: pd.DataFrame) -> list[str]:
    """값 형식이 '불규칙 소수'인 대상자 (출처 확인이 필요한 라벨)."""
    return sorted(subjects.loc[subjects["value_format"] == "irregular", "subject_id"].tolist())


# ----------------------------------------------------------------- descriptives


def _describe(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {"n": 0, **{k: np.nan for k in ("mean", "median", "std", "min", "q1", "q3", "max", "iqr")}}
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    return {
        "n": int(x.size),
        "mean": x.mean(),
        "median": x.median(),
        "std": x.std(ddof=1) if x.size > 1 else np.nan,
        "min": x.min(),
        "q1": q1,
        "q3": q3,
        "max": x.max(),
        "iqr": q3 - q1,
    }


def summarize_scores(df: pd.DataFrame, by: list[str] | None = None, scores=SCORES, unit: str = "Session") -> pd.DataFrame:
    """점수별 기술통계. 결측은 제외하고 n 은 실제 사용된 관측 수다."""
    by = list(by or [])
    rows = []
    groups = df.groupby(by, sort=False) if by else [((), df)]
    for key, grp in groups:
        key = key if isinstance(key, tuple) else (key,)
        for s in scores:
            rows.append({"unit": unit, **dict(zip(by, key)), "score": s, **_describe(grp[s])})
    out = pd.DataFrame(rows)
    if "group" in by:
        out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)
        out = out.sort_values([*by, "score"]).reset_index(drop=True)
        out["group"] = out["group"].astype(str)
    return out


def describe_by(df: pd.DataFrame, value: str, by: str, order=None) -> pd.DataFrame:
    rows = []
    keys = order or list(dict.fromkeys(df[by]))
    for k in keys:
        rows.append({by: k, **_describe(df.loc[df[by] == k, value])})
    rows.append({by: "전체", **_describe(df[value])})
    return pd.DataFrame(rows)


def check_score_ranges(sessions: pd.DataFrame) -> pd.DataFrame:
    """정의된 범위(TS 0~50, PO 0~15, CF 0~35)를 벗어난 관측. 결측은 범위 검사 대상이 아니다."""
    rows = []
    for s in SCORES:
        lo, hi = labels.RANGES[s]
        bad = sessions[sessions[s].notna() & ((sessions[s] < lo) | (sessions[s] > hi))]
        for r in bad.itertuples():
            rows.append({"subject_id": r.subject_id, "exercise": r.exercise, "score": s, "value": getattr(r, s)})
    return pd.DataFrame(rows, columns=["subject_id", "exercise", "score", "value"])


def observed_ranges(sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SCORES:
        lo, hi = labels.RANGES[s]
        x = sessions[s].dropna()
        rows.append({"score": s, "defined_min": lo, "defined_max": hi, "observed_min": x.min(),
                     "observed_max": x.max(), "n": int(x.size), "n_missing": int(sessions[s].isna().sum()),
                     "n_out_of_range": int(((x < lo) | (x > hi)).sum())})
    return pd.DataFrame(rows)


def sum_residual(sessions: pd.DataFrame, tol: float = labels.SUM_TOL) -> pd.DataFrame:
    """TS − (PO + CF). 세 점수가 모두 있는 Session 만 계산한다."""
    df = sessions.dropna(subset=list(SCORES)).copy()
    df["ts_minus_sum"] = df["TS"] - (df["PO"] + df["CF"])
    df["nonzero"] = df["ts_minus_sum"].abs() > tol
    return df[["subject_id", "group", "exercise", "TS", "PO", "CF", "ts_minus_sum", "nonzero"]]


def ceiling_table(sessions: pd.DataFrame, score: str, by: str | None = None, frac: float = NEAR_CEILING_FRAC) -> pd.DataFrame:
    """상한 근처(만점의 frac 이상)와 만점 관측 비율."""
    hi = labels.RANGES[score][1]
    thr = hi * frac
    keys = [None] if by is None else list(dict.fromkeys(sessions[by]))
    if by == "group":
        keys = [g for g in GROUP_ORDER if g in set(sessions[by])]
    rows = []
    for k in keys:
        x = (sessions if k is None else sessions[sessions[by] == k])[score].dropna()
        rows.append({
            (by or "subset"): "전체" if k is None else k,
            "score": score,
            "threshold": thr,
            "n": int(x.size),
            "n_near_ceiling": int((x >= thr).sum()),
            "share_near_ceiling": float((x >= thr).mean()) if x.size else np.nan,
            "n_at_max": int((x >= hi - 1e-9).sum()),
            "share_at_max": float((x >= hi - 1e-9).mean()) if x.size else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- relationships


def correlation(x, y) -> dict:
    """Pearson·Spearman. 결측 쌍은 제외. n<3 이면 NaN."""
    d = pd.DataFrame({"x": pd.to_numeric(pd.Series(list(x)), errors="coerce"),
                      "y": pd.to_numeric(pd.Series(list(y)), errors="coerce")}).dropna()
    n = len(d)
    out = {"n": n, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    if n >= 3 and d["x"].nunique() > 1 and d["y"].nunique() > 1:
        pr = stats.pearsonr(d["x"], d["y"])
        sr = stats.spearmanr(d["x"], d["y"])
        out.update(pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
                   spearman_rho=float(sr.statistic), spearman_p=float(sr.pvalue))
    return out


def po_cf_relation(sessions: pd.DataFrame) -> pd.DataFrame:
    """PO–CF 관계 (Session 단위; 같은 대상자의 운동 5개가 섞여 있어 p-value 는 참고용)."""
    rows = [{"subset": "전체", **correlation(sessions["PO"], sessions["CF"])}]
    for g in GROUP_ORDER:
        sub = sessions[sessions["group"] == g]
        if len(sub):
            rows.append({"subset": g, **correlation(sub["PO"], sub["CF"])})
    return pd.DataFrame(rows)


def age_score_correlation(sessions: pd.DataFrame) -> pd.DataFrame:
    """Age 와 점수의 Spearman. Subject 단위(운동 평균)와 운동별(운동마다 대상자 1명=1관측)로 계산한다.

    코호트(건강군/환자군) 안에서도 계산해 Group 차이와 Age 효과가 섞이는 정도를 본다.
    """
    subj = subject_scores(sessions)
    rows = []
    for cohort in ("전체", *COHORT_ORDER):
        s_sub = subj if cohort == "전체" else subj[subj["cohort"] == cohort]
        x_sub = sessions if cohort == "전체" else sessions[sessions["cohort"] == cohort]
        for s in SCORES:
            c = correlation(s_sub["age"], s_sub[s])
            rows.append({"cohort": cohort, "unit": "Subject(운동 평균)", "exercise": "평균", "score": s,
                         "n": c["n"], "rho": c["spearman_rho"], "p": c["spearman_p"]})
            for ex in labels.EXERCISES:
                e = x_sub[x_sub["exercise"] == ex]
                c = correlation(e["age"], e[s])
                rows.append({"cohort": cohort, "unit": "Session(운동별)", "exercise": ex, "score": s,
                             "n": c["n"], "rho": c["spearman_rho"], "p": c["spearman_p"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ SuppInfo/Age


def suppinfo_completeness(subjects: pd.DataFrame) -> dict:
    """Subject 수와 Age·점수 확보 수를 구분해 센다."""
    per_group = []
    for g in GROUP_ORDER:
        s = subjects[subjects["group"] == g]
        if s.empty:
            continue
        per_group.append({
            "group": g,
            "subjects": len(s),
            "suppinfo": int(s["has_suppinfo"].sum()),
            "age": int(s["has_age"].sum()),
            "gender": int(s["gender"].notna().sum()) if "gender" in s else 0,
            "clinical_C1": int(s["C1_has_clinical_file"].sum()),
            "clinical_C2": int(s["C2_has_any_score"].sum()),
            "age_and_C2": int((s["has_age"] & s["C2_has_any_score"]).sum()),
        })
    table = pd.DataFrame(per_group)
    total = table.drop(columns="group").sum()
    table = pd.concat([table, pd.DataFrame([{"group": "합계", **total.to_dict()}])], ignore_index=True)
    n = len(subjects)
    missing_rate = {
        col: float(1 - subjects[col].notna().mean()) for col in ("age", "gender", "supp_group") if col in subjects
    }
    return {
        "table": table,
        "n_subjects": n,
        "n_age": int(subjects["has_age"].sum()),
        "n_clinical_C1": int(subjects["C1_has_clinical_file"].sum()),
        "n_clinical_C2": int(subjects["C2_has_any_score"].sum()),
        "n_age_and_C2": int((subjects["has_age"] & subjects["C2_has_any_score"]).sum()),
        "age_missing_subjects": sorted(subjects.loc[~subjects["has_age"], "subject_id"].tolist()),
        "missing_rate": missing_rate,
    }


def age_overlap(subjects: pd.DataFrame, bin_width: int = AGE_BIN_WIDTH, caliper: float = MATCH_CALIPER) -> dict:
    """건강군·환자군 연령 overlap 과 1:1 연령 매칭 가능 수 (Age 가 있는 대상자만)."""
    s = subjects[subjects["has_age"]]
    h = np.sort(s.loc[s["cohort"] == COHORT_ORDER[0], "age"].to_numpy(dtype=float))
    p = np.sort(s.loc[s["cohort"] == COHORT_ORDER[1], "age"].to_numpy(dtype=float))
    res = {"n_healthy": len(h), "n_patient": len(p)}
    if len(h) == 0 or len(p) == 0:
        return {**res, "class": "C", "reason": "한 군의 Age 가 없음"}

    lo, hi = max(h.min(), p.min()), min(h.max(), p.max())
    in_h = ((h >= lo) & (h <= hi)) if lo <= hi else np.zeros_like(h, dtype=bool)
    in_p = ((p >= lo) & (p <= hi)) if lo <= hi else np.zeros_like(p, dtype=bool)

    start = int(np.floor(min(h.min(), p.min()) / bin_width) * bin_width)
    stop = int(np.ceil((max(h.max(), p.max()) + 1) / bin_width) * bin_width)
    edges = np.arange(start, stop + bin_width, bin_width)
    hc, _ = np.histogram(h, edges)
    pc, _ = np.histogram(p, edges)
    bins = pd.DataFrame({"age_bin": [f"{a}–{a + bin_width - 1}" for a in edges[:-1]],
                         COHORT_ORDER[0]: hc, COHORT_ORDER[1]: pc})
    bins = bins[(bins[COHORT_ORDER[0]] + bins[COHORT_ORDER[1]]) > 0].reset_index(drop=True)
    n_bins_both = int(((bins[COHORT_ORDER[0]] >= OVERLAP_MIN_PER_BIN) & (bins[COHORT_ORDER[1]] >= OVERLAP_MIN_PER_BIN)).sum())

    # 1:1 탐욕 매칭: 환자를 나이 순으로, 남은 건강군 중 가장 가까운 사람 (|차이| ≤ caliper)
    pool = list(h)
    pairs = []
    for a in p:
        if not pool:
            break
        j = int(np.argmin([abs(a - b) for b in pool]))
        if abs(a - pool[j]) <= caliper:
            pairs.append((a, pool.pop(j)))

    frac_h, frac_p = float(in_h.mean()), float(in_p.mean())
    if lo > hi or min(frac_h, frac_p) < 0.25:
        cls = "C"
    elif min(frac_h, frac_p) >= 0.75 and n_bins_both >= 3:
        cls = "A"
    else:
        cls = "B"
    return {
        **res,
        "healthy_min": float(h.min()), "healthy_max": float(h.max()),
        "patient_min": float(p.min()), "patient_max": float(p.max()),
        "common_low": float(lo) if lo <= hi else np.nan,
        "common_high": float(hi) if lo <= hi else np.nan,
        "n_healthy_in_common": int(in_h.sum()), "n_patient_in_common": int(in_p.sum()),
        "frac_healthy_in_common": frac_h, "frac_patient_in_common": frac_p,
        "bins": bins, "n_bins_both": n_bins_both,
        "n_matched_pairs": len(pairs), "caliper": caliper,
        "matched_mean_age_diff": float(np.mean([a - b for a, b in pairs])) if pairs else np.nan,
        "class": cls,
    }


# -------------------------------------------------------------------- tests


def kruskal_test(df: pd.DataFrame, value: str, by: str, order) -> dict:
    groups = [df.loc[df[by] == g, value].dropna().to_numpy() for g in order]
    groups = [g for g in groups if len(g)]
    ns = [len(g) for g in groups]
    out = {"test": "Kruskal-Wallis", "value": value, "groups": len(groups), "n": int(sum(ns)),
           "min_group_n": int(min(ns)) if ns else 0, "statistic": np.nan, "p": np.nan,
           "effect": "epsilon²", "effect_size": np.nan}
    if len(groups) >= 2 and all(n >= 1 for n in ns) and sum(ns) > len(groups):
        h, pval = stats.kruskal(*groups)
        n, k = sum(ns), len(groups)
        out.update(statistic=float(h), p=float(pval), effect_size=float((h - k + 1) / (n - k)))
    out["unstable"] = out["min_group_n"] < MIN_N_STABLE
    return out


def mannwhitney_test(df: pd.DataFrame, value: str, by: str, a: str, b: str) -> dict:
    """양측 Mann-Whitney U. 효과크기 rank-biserial r = 2·U_a/(n_a·n_b) − 1 (양수 = a 가 큼)."""
    xa = df.loc[df[by] == a, value].dropna().to_numpy()
    xb = df.loc[df[by] == b, value].dropna().to_numpy()
    out = {"test": f"Mann-Whitney ({a} vs {b})", "value": value, "groups": 2, "n": len(xa) + len(xb),
           "min_group_n": min(len(xa), len(xb)), "statistic": np.nan, "p": np.nan,
           "effect": "rank-biserial r", "effect_size": np.nan}
    if len(xa) and len(xb):
        res = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        out.update(statistic=float(res.statistic), p=float(res.pvalue),
                   effect_size=float(2 * res.statistic / (len(xa) * len(xb)) - 1))
    out["unstable"] = out["min_group_n"] < MIN_N_STABLE
    return out


def group_tests(subjects: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Subject 단위 검정: 점수(운동 평균)와 Age 의 그룹 차이."""
    subj = subject_scores(sessions)
    rows = []
    for v in SCORES:
        rows.append(kruskal_test(subj, v, "group", GROUP_ORDER))
        rows.append(mannwhitney_test(subj, v, "cohort", *COHORT_ORDER))
        rows.append(mannwhitney_test(subj, v, "group", "Expert", "NotExpert"))
    age = subjects[subjects["has_age"]]
    rows.append(kruskal_test(age, "age", "group", GROUP_ORDER))
    rows.append(mannwhitney_test(age, "age", "cohort", *COHORT_ORDER))
    out = pd.DataFrame(rows)
    out["unit"] = "Subject"
    return out


# --------------------------------------------------------------- sensitivity


def compare_label_sensitivity(subjects: pd.DataFrame, sessions: pd.DataFrame, flagged: list[str]) -> dict:
    """불규칙 소수 라벨 대상자 포함(A) vs 제외(B)."""
    a_sess, b_sess = sessions, sessions[~sessions["subject_id"].isin(flagged)]
    a_subj, b_subj = subjects, subjects[~subjects["subject_id"].isin(flagged)]

    def by_group(sess):
        return summarize_scores(sess, ["group"])[["group", "score", "n", "mean", "median"]]

    ga, gb = by_group(a_sess), by_group(b_sess)
    grp = ga.merge(gb, on=["group", "score"], how="outer", suffixes=("_A", "_B"))
    grp["mean_diff_B_minus_A"] = grp["mean_B"] - grp["mean_A"]
    grp["median_diff_B_minus_A"] = grp["median_B"] - grp["median_A"]

    def key_corrs(sess):
        subj = subject_scores(sess)
        rows = [{"relation": "PO vs CF (Session)", **correlation(sess["PO"], sess["CF"])}]
        for s in SCORES:
            rows.append({"relation": f"Age vs {s} (Subject 평균)", **correlation(subj["age"], subj[s])})
        return pd.DataFrame(rows)[["relation", "n", "spearman_rho", "spearman_p"]]

    corr = key_corrs(a_sess).merge(key_corrs(b_sess), on="relation", suffixes=("_A", "_B"))
    corr["rho_diff_B_minus_A"] = corr["spearman_rho_B"] - corr["spearman_rho_A"]

    tests_a = group_tests(a_subj, a_sess)
    tests_b = group_tests(b_subj, b_sess)
    tests = tests_a[["test", "value", "n", "p", "effect_size"]].merge(
        tests_b[["test", "value", "n", "p", "effect_size"]], on=["test", "value"], suffixes=("_A", "_B"))

    return {
        "flagged": flagged,
        "flagged_groups": subjects.loc[subjects["subject_id"].isin(flagged), ["subject_id", "group"]],
        "n_sessions": (len(a_sess), len(b_sess)),
        "n_subjects": (a_sess["subject_id"].nunique(), b_sess["subject_id"].nunique()),
        "overall": summarize_scores(a_sess).merge(summarize_scores(b_sess), on=["unit", "score"], suffixes=("_A", "_B")),
        "by_group": grp,
        "correlations": corr,
        "tests": tests,
    }


# -------------------------------------------------------------------- report


def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "예" if v else "아니오"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        if abs(v) != 0 and (abs(v) < 0.001):
            return f"{v:.1e}"
        return f"{v:.3f}".rstrip("0").rstrip(".") if not float(v).is_integer() else str(int(v))
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or len(df) == 0:
        return "- 해당 없음"
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def analyze_age_confounding(subjects: pd.DataFrame, sessions: pd.DataFrame) -> dict:
    """Age confounding 묶음: 완결성, Group·코호트별 Age, Age–점수 Spearman, 연령 overlap 판정.

    반환값은 DataFrame / dict 이며 화면·파일 출력에 의존하지 않는다.
    """
    has_age = subjects[subjects["has_age"]]
    present = [g for g in GROUP_ORDER if g in set(subjects["group"])]
    return {
        "completeness": suppinfo_completeness(subjects),
        "age_by_group": describe_by(has_age, "age", "group", present),
        "age_by_cohort": describe_by(has_age, "age", "cohort", list(COHORT_ORDER)),
        "age_score_spearman": age_score_correlation(sessions),
        "overlap": age_overlap(subjects),
    }


def analyze_label_sensitivity(subjects: pd.DataFrame, sessions: pd.DataFrame) -> dict:
    """출처 확인이 필요한 라벨(불규칙 소수) 포함(A) vs 제외(B) 비교."""
    return compare_label_sensitivity(subjects, sessions, irregular_subjects(subjects))


def run_analysis(raw_root: str | Path) -> dict:
    subjects, sessions = load_label_tables(raw_root)
    return analyze(subjects, sessions)


def analyze(subjects: pd.DataFrame, sessions: pd.DataFrame) -> dict:
    """Subject 표와 Session 표로 Gate 2 분석 전체를 수행한다."""
    flagged = irregular_subjects(subjects)
    age = analyze_age_confounding(subjects, sessions)
    resid = sum_residual(sessions)
    return {
        "subjects": subjects,
        "sessions": sessions,
        "subject_scores": subject_scores(sessions),
        "flagged": flagged,
        "supp": age["completeness"],
        "ranges": observed_ranges(sessions),
        "range_violations": check_score_ranges(sessions),
        "summary_group": summarize_scores(sessions, ["group"]),
        "summary_group_ex": summarize_scores(sessions, ["group", "exercise"]),
        "summary_exercise": summarize_scores(sessions, ["exercise"]),
        "summary_subject": summarize_scores(subject_scores(sessions), ["group"], unit="Subject(운동 평균)"),
        "residual": resid,
        "po_cf": po_cf_relation(sessions),
        "ceiling": {s: {k: ceiling_table(sessions, s, by=k) for k in (None, "group", "exercise", "cohort")} for s in SCORES},
        "age_desc": age["age_by_group"],
        "age_desc_cohort": age["age_by_cohort"],
        "age_corr": age["age_score_spearman"],
        "overlap": age["overlap"],
        "tests": group_tests(subjects, sessions),
        "sensitivity": analyze_label_sensitivity(subjects, sessions),
    }


def main(argv: list[str] | None = None) -> None:
    from kimore import label_eda_plots, label_eda_report

    ap = argparse.ArgumentParser(description="Gate 2 라벨·Age EDA (원본은 읽기만 함)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--report", default="docs/label_eda_report.md")
    ap.add_argument("--fig-dir", default="docs/figures/label_eda")
    args = ap.parse_args(argv)
    res = run_analysis(args.raw)
    report = Path(args.report)
    figs = label_eda_plots.make_figures(res, Path(args.fig_dir))
    label_eda_report.write_report(res, figs, report, generated=date.today().isoformat())
    sup = res["supp"]
    print(f"subjects={sup['n_subjects']} age={sup['n_age']} clinical_C1={sup['n_clinical_C1']} "
          f"sessions={len(res['sessions'])} flagged={len(res['flagged'])} overlap_class={res['overlap']['class']} "
          f"figures={len(figs)} report={report}")


if __name__ == "__main__":
    main()
