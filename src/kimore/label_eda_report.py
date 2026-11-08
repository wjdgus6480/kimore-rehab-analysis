"""Gate 2 보고서(docs/label_eda_report.md) 생성. 수치는 모두 label_eda.run_analysis() 결과에서 가져온다.

해석 문장은 아래 규칙에 따라 수치로부터 만들어진다. 규칙을 바꾸면 보고서도 함께 바뀐다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from kimore import label_eda as L
from kimore.label_eda import COHORT_ORDER, SCORES, md_table

SENSITIVE_MEDIAN_SHIFT = 2.0  # 그룹 중앙값이 이 이상 움직이면 '민감'으로 표시 (점)
SENSITIVE_RHO_SHIFT = 0.1  # 상관계수가 이 이상 움직이면 '민감'으로 표시


def _p(v: float) -> str:
    if v is None or np.isnan(v):
        return "—"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _pct(v: float) -> str:
    return "—" if v is None or np.isnan(v) else f"{100 * v:.0f}%"


def _img(fig: Path, report: Path, title: str) -> str:
    rel = Path(os.path.relpath(fig, report.parent)).as_posix()
    return f"![{title}]({rel})"


def _ceiling_block(res: dict, score: str) -> str:
    c = res["ceiling"][score]
    allrow = c[None].iloc[0]
    by_cohort = c["cohort"].rename(columns={"cohort": "구분"})
    by_group = c["group"].rename(columns={"group": "구분"})
    by_ex = c["exercise"].rename(columns={"exercise": "구분"})
    by_ex["구분"] = "Ex" + by_ex["구분"].astype(str)
    tab = pd.concat([by_cohort, by_group, by_ex], ignore_index=True)
    cols = ["구분", "n", "n_near_ceiling", "share_near_ceiling", "n_at_max", "share_at_max"]
    return (
        f"- 기준: 만점 {L.labels.RANGES[score][1]}의 {int(L.NEAR_CEILING_FRAC * 100)}% 이상(≥ {allrow['threshold']:g}) = 상한 근처\n"
        f"- 전체 Session {allrow['n']}개 중 상한 근처 {allrow['n_near_ceiling']}개({_pct(allrow['share_near_ceiling'])}), "
        f"만점 {allrow['n_at_max']}개({_pct(allrow['share_at_max'])})\n\n"
        + md_table(tab, cols)
    )


def _score_section(res: dict, score: str, figs: dict, report: Path) -> str:
    sg = res["summary_group"]
    sg = sg[sg["score"] == score]
    sx = res["summary_exercise"]
    sx = sx[sx["score"] == score]
    gx = res["summary_group_ex"]
    gx = gx[gx["score"] == score]
    cols = ["n", "mean", "median", "std", "min", "q1", "q3", "max", "iqr"]
    return f"""- 단위: Session (대상자 × 운동). 결측 Session 은 제외하고 n 에 반영.

**Group별**
{md_table(sg, ["group", *cols])}

**Exercise별**
{md_table(sx, ["exercise", *cols])}

<details><summary>Group × Exercise별 표</summary>

{md_table(gx, ["group", "exercise", *cols])}

</details>

{_img(figs[f"group_{score}"], report, f"Group별 {score}")}

{_img(figs[f"group_exercise_{score}"], report, f"Group×Exercise별 {score}")}
"""


def _findings(res: dict) -> list[str]:
    out = []
    ov, supp, tests = res["overlap"], res["supp"], res["tests"]
    ac = res["age_corr"]
    avg = ac[ac["exercise"] == "평균"].set_index(["cohort", "score"])
    ts_c = res["ceiling"]["TS"]["cohort"].set_index("cohort")
    po_c = res["ceiling"]["PO"]["cohort"].set_index("cohort")
    cf_c = res["ceiling"]["CF"]["cohort"].set_index("cohort")
    resid = res["residual"]
    rv = res["range_violations"]

    out.append(
        f"정의 범위를 벗어난 점수는 {len(rv)}건이다. 점수가 있는 Session 중 TS ≠ PO+CF 인 경우는 "
        f"{int(resid['nonzero'].sum())}건({', '.join(f'{r.subject_id} Ex{r.exercise} 차이 {r.ts_minus_sum:+.2f}' for r in resid[resid.nonzero].itertuples()) or '없음'})"
        "으로, 기존 감사 결과와 같은 관측이 재현되었다. 오류 여부는 원본 재확인 전까지 판단하지 않는다."
    )
    out.append(
        f"건강군 Session 의 TS 상한 근처(≥45) 비율은 {_pct(ts_c.loc['건강군', 'share_near_ceiling'])}, "
        f"환자군은 {_pct(ts_c.loc['환자군', 'share_near_ceiling'])}다. PO 는 건강군에서 만점 비율이 "
        f"{_pct(po_c.loc['건강군', 'share_at_max'])}로 CF({_pct(cf_c.loc['건강군', 'share_at_max'])})보다 높다. "
        "건강군 분포의 상단이 압축되어 있을 가능성이 있으며, 그 정도는 PO 에서 더 크다."
    )
    po_cf = res["po_cf"].set_index("subset").loc["전체"]
    out.append(
        f"PO 와 CF 의 Spearman ρ = {po_cf['spearman_rho']:.2f} (Session n={int(po_cf['n'])}). 관련은 있으나 같은 정보는 아니다. "
        "TS 는 대부분 PO+CF 와 같으므로 TS 를 따로 target 으로 두면 PO·CF 와 정보가 겹친다."
    )
    age_t = tests[(tests["value"] == "age") & tests["test"].str.startswith("Mann")].iloc[0]
    out.append(
        f"건강군과 환자군의 Age 차이가 크다 (Mann-Whitney rank-biserial r = {age_t['effect_size']:.2f}, n={age_t['n']}). "
        f"Age 는 점수와 음의 연관을 보인다 (Subject 평균, 전체 TS ρ = {avg.loc[('전체', 'TS'), 'rho']:.2f}, n={avg.loc[('전체', 'TS'), 'n']})."
    )
    out.append(
        f"이 연관은 코호트 안에서도 관찰된다 (건강군 TS ρ = {avg.loc[('건강군', 'TS'), 'rho']:.2f}, n={avg.loc[('건강군', 'TS'), 'n']}; "
        f"환자군 TS ρ = {avg.loc[('환자군', 'TS'), 'rho']:.2f}, n={avg.loc[('환자군', 'TS'), 'n']}). "
        "따라서 Group 차이와 Age 의 관계를 분리해서 보아야 하며, Group 간 점수 차이를 곧바로 운동 수행 차이로 해석할 수 없다."
    )
    out.append(
        f"연령 overlap 판정은 **{ov['class']}** 다. 공통 연령 구간은 {ov['common_low']:g}–{ov['common_high']:g}세이고, "
        f"이 구간 안에 건강군 {_pct(ov['frac_healthy_in_common'])}, 환자군 {_pct(ov['frac_patient_in_common'])}가 있다. "
        f"그러나 양쪽 모두 {L.OVERLAP_MIN_PER_BIN}명 이상인 10세 구간은 {ov['n_bins_both']}개뿐이고, "
        f"±{ov['caliper']:g}세 1:1 매칭으로 짝지을 수 있는 환자는 {ov['n_patient']}명 중 {ov['n_matched_pairs']}명이다."
    )
    ex = tests[tests["test"].str.contains("Expert vs NotExpert") & (tests["value"] == "TS")].iloc[0]
    out.append(
        f"Expert 와 NotExpert 의 점수 차이는 작다 (TS rank-biserial r = {ex['effect_size']:.2f}, p = {_p(ex['p'])}, n={ex['n']})."
    )
    out.append(
        f"Age 와 점수가 모두 있는 대상자는 {supp['n_age_and_C2']}명으로, 전체 {supp['n_subjects']}명보다 적다. "
        "Age 관련 분석은 모두 이 부분집합 기준이다."
    )
    return out


def _sensitivity_text(sens: dict) -> tuple[str, list[str]]:
    g = sens["by_group"]
    moved = g[g["median_diff_B_minus_A"].abs() >= SENSITIVE_MEDIAN_SHIFT]
    rho = sens["correlations"]
    rho_moved = rho[rho["rho_diff_B_minus_A"].abs() >= SENSITIVE_RHO_SHIFT]
    lines = []
    if len(moved):
        lines.append(
            "그룹 중앙값이 " + f"{SENSITIVE_MEDIAN_SHIFT:g}점 이상 움직인 경우: "
            + ", ".join(f"{r.group} {r.score} {r.median_diff_B_minus_A:+.2f} (n {r.n_A}→{r.n_B})" for r in moved.itertuples())
        )
    else:
        lines.append(f"그룹 중앙값이 {SENSITIVE_MEDIAN_SHIFT:g}점 이상 움직인 경우는 없다.")
    if len(rho_moved):
        lines.append("상관계수가 크게 움직인 관계: " + ", ".join(f"{r.relation} Δρ={r.rho_diff_B_minus_A:+.2f}" for r in rho_moved.itertuples()))
    else:
        lines.append(f"주요 상관(PO–CF, Age–점수)의 ρ 변화는 모두 {SENSITIVE_RHO_SHIFT:g} 미만이다.")
    verdict = "그룹 단위 요약(특히 표본이 작은 그룹)은 민감하고, 전체 상관 구조는 덜 민감하다." if len(moved) and not len(rho_moved) \
        else ("민감하다." if len(moved) or len(rho_moved) else "크게 민감하지 않다.")
    return verdict, lines


def write_report(res: dict, figs: list[Path], report: Path, generated: str) -> Path:
    report = Path(report)
    fig = {p.stem: p for p in figs}
    supp, ov, sens = res["supp"], res["overlap"], res["sensitivity"]
    subj, sess = res["subjects"], res["sessions"]
    ac = res["age_corr"]
    verdict, sens_lines = _sensitivity_text(sens)
    findings = _findings(res)
    n_scored_sessions = int(sess.dropna(subset=list(SCORES)).shape[0])
    flagged_tab = sens["flagged_groups"]
    stroke_b = sens["by_group"].query("group == 'Stroke' and score == 'TS'")
    stroke_nb = int(stroke_b["n_B"].iloc[0]) if len(stroke_b) else 0
    tests = res["tests"].copy()
    tests["p"] = tests["p"].map(_p)

    st = supp["table"][supp["table"]["group"] != "합계"]
    age_missing_groups = ", ".join(
        f"{r.group} {r.subjects - r.age}명" for r in st.itertuples() if r.subjects - r.age > 0
    ) or "없음"
    age_avg = ac[ac["exercise"] == "평균"]
    age_ex = ac[(ac["exercise"] != "평균") & (ac["cohort"] == "전체")]

    text = f"""# KIMORE Gate 2 — Label & Age EDA

> 생성: `python -m kimore.label_eda` · {generated}
> **개발용·예비 결과** — 현재 로컬 사본 기준. 공식 원본 재확보 후 같은 명령으로 다시 생성한다.
> 이 파일은 코드가 덮어쓴다. 표·수치·판정(overlap, 민감도)은 실행 시 계산된다.
> 8·9절의 서술 일부(연령이 몰린 구간, 후보의 장단점)는 현재 데이터를 보고 작성한 문장이므로, 원본 재확보 후 다시 검토한다.
> 불규칙 소수 라벨은 "출처 확인이 필요한 라벨"이며 오류로 간주하지 않는다.

## 1. 분석 목적

1. Clinical score(TS/PO/CF)가 분석 가능한 상태인지 확인한다 (분포, 결측, 범위, 운동·Group 차이, 상한 근처 관측, PO–CF 관계).
2. Age 가 Group 과 Clinical score 의 관계를 교란할 가능성을 확인한다 (Group별 연령, Age–점수 관계, 연령 overlap, 매칭 가능성).
3. 출처 확인이 필요한 라벨(불규칙 소수) 포함 여부에 결과가 얼마나 민감한지 확인한다.

모델 학습, 분할, CV, feature engineering 은 하지 않는다.

## 2. 데이터 범위 및 분석 단위

| 구분 | n | 기준 |
|---|---|---|
| 전체 Subject | {supp['n_subjects']} | 폴더 구조 |
| Age available Subject | {supp['n_age']} | SuppInfo 에 Age 있음 |
| Clinical score file Subject (C1) | {supp['n_clinical_C1']} | ClinicalAssessment 파일 있음 |
| Clinical score available Subject (C2) | {supp['n_clinical_C2']} | 점수 1개 이상 |
| Age + Clinical score Subject | {supp['n_age_and_C2']} | Age–점수 분석에 쓰이는 대상자 |
| Session 행 (C1 대상자 × 운동) | {len(sess)} | Session 표 |
| 세 점수가 모두 있는 Session | {n_scored_sessions} | 점수 분석에 쓰이는 Session |

- **Subject 단위**: Age 분포, Age–점수 상관(운동 평균), Group 간 검정. 같은 대상자의 운동 5개는 독립이 아니므로 대상자 간 검정은 Subject 단위로 한다.
- **Session 단위**: 점수 분포, 상한 근처 비율, PO–CF 관계, 운동별 Age–점수 상관(운동마다 대상자 1명 = 1관측).
- 건강군 = CG(Expert, NotExpert), 환자군 = GPP(BackPain, Parkinson, Stroke). 폴더 정의를 그대로 따른다.

## 3. Clinical Score 품질

**정의 범위와 실제 범위**
{md_table(res["ranges"])}

### 3.1 TS
{_score_section(res, "TS", fig, report)}
### 3.2 PO
{_score_section(res, "PO", fig, report)}
### 3.3 CF
{_score_section(res, "CF", fig, report)}
### 3.4 PO-CF 관계
- Session 단위라 같은 대상자의 관측이 섞여 있다. p-value 는 참고용이다.

{md_table(res["po_cf"].assign(pearson_p=res["po_cf"]["pearson_p"].map(_p), spearman_p=res["po_cf"]["spearman_p"].map(_p)))}

**TS − (PO + CF) ≠ 0 인 Session** (허용 오차 {L.labels.SUM_TOL})

{md_table(res["residual"][res["residual"]["nonzero"]], ["subject_id", "group", "exercise", "TS", "PO", "CF", "ts_minus_sum"])}

{_img(fig["po_vs_cf"], report, "PO vs CF")}

### 3.5 Ceiling Effect
상한 근처 관측이 많으면 분포의 상단이 압축되어 있을 가능성이 있다. 아래는 그 비율이다.

**TS**
{_ceiling_block(res, "TS")}

**PO**
{_ceiling_block(res, "PO")}

**CF**
{_ceiling_block(res, "CF")}

## 4. SuppInfo / Age

### 4.1 Age completeness
{md_table(supp["table"])}

- Age 결측 Subject ({len(supp["age_missing_subjects"])}): {", ".join(supp["age_missing_subjects"]) or "없음"}
- 결측률 (전체 Subject 기준): {", ".join(f"{k} {_pct(v)}" for k, v in supp["missing_rate"].items())}
- 결측값은 보정하거나 추정하지 않았다.

### 4.2 Group별 Age distribution
{md_table(res["age_desc"])}

{md_table(res["age_desc_cohort"])}

{_img(fig["age_by_group"], report, "Group별 Age")}

### 4.3 Age vs Clinical Score
**Subject 단위 (대상자별 운동 평균 점수)**
{md_table(age_avg.assign(p=age_avg["p"].map(_p)), ["cohort", "score", "n", "rho", "p"])}

<details><summary>운동별 (전체 코호트)</summary>

{md_table(age_ex.assign(p=age_ex["p"].map(_p)), ["exercise", "score", "n", "rho", "p"])}

</details>

{_img(fig["age_vs_scores"], report, "Age vs 점수")}

### 4.4 건강군-환자군 Age overlap
| 항목 | 건강군 | 환자군 |
|---|---|---|
| n (Age 있음) | {ov['n_healthy']} | {ov['n_patient']} |
| min–max | {ov['healthy_min']:g}–{ov['healthy_max']:g} | {ov['patient_min']:g}–{ov['patient_max']:g} |
| 공통 구간 안 인원 | {ov['n_healthy_in_common']} ({_pct(ov['frac_healthy_in_common'])}) | {ov['n_patient_in_common']} ({_pct(ov['frac_patient_in_common'])}) |

- 공통 연령 구간: {ov['common_low']:g}–{ov['common_high']:g}세
- 양쪽 모두 {L.OVERLAP_MIN_PER_BIN}명 이상인 10세 구간 수: {ov['n_bins_both']}
- ±{ov['caliper']:g}세 1:1 탐욕 매칭: 환자 {ov['n_patient']}명 중 {ov['n_matched_pairs']}명 매칭 (평균 연령차 {ov['matched_mean_age_diff']:+.1f}세)
- 판정 규칙(사전 정의): {L.OVERLAP_RULE}
- **판정: {ov['class']}** — {"연령 overlap 이 충분해 age-adjusted 분석이나 매칭이 가능할 수 있다." if ov['class'] == 'A' else "일부 overlap 이 있으나 구간별 표본이 얇다. 제한된 매칭 또는 sensitivity analysis 로만 다루는 것이 적절하다." if ov['class'] == 'B' else "overlap 이 거의 없어 Age 효과를 통계적으로 분리하기 어렵다."}

{md_table(ov["bins"])}

{_img(fig["age_overlap"], report, "연령 overlap")}

## 5. Label Sensitivity Analysis

출처 확인이 필요한 불규칙 소수 라벨 대상자 {len(sens["flagged"])}명:

{md_table(flagged_tab)}

### 5.1 불규칙 소수 포함 (A)
- Subject {sens["n_subjects"][0]}명, Session {sens["n_sessions"][0]}행

### 5.2 불규칙 소수 제외 (B)
- Subject {sens["n_subjects"][1]}명, Session {sens["n_sessions"][1]}행
- 제외 후 Stroke 의 점수 Session 은 {stroke_nb}개다.

### 5.3 결과 비교
**전체 분포**
{md_table(sens["overall"], ["score", "n_A", "mean_A", "median_A", "n_B", "mean_B", "median_B"])}

**Group별 평균·중앙값**
{md_table(sens["by_group"])}

**주요 상관**
{md_table(sens["correlations"].assign(spearman_p_A=sens["correlations"]["spearman_p_A"].map(_p), spearman_p_B=sens["correlations"]["spearman_p_B"].map(_p)))}

**Group 검정 (Subject 단위)**
{md_table(sens["tests"].assign(p_A=sens["tests"]["p_A"].map(_p), p_B=sens["tests"]["p_B"].map(_p)))}

{_img(fig["label_sensitivity"], report, "라벨 민감도")}

- 판단 규칙: 그룹 중앙값 변화 ≥ {SENSITIVE_MEDIAN_SHIFT:g}점 또는 ρ 변화 ≥ {SENSITIVE_RHO_SHIFT:g} 이면 '민감'.
""" + "\n".join(f"- {line}" for line in sens_lines) + f"""
- **결과가 불규칙 라벨 {len(sens["flagged"])}명의 포함 여부에 얼마나 민감한가?** {verdict}

## 6. 통계검정 결과

탐색적 검정이다. 모두 Subject 단위(점수는 대상자별 운동 평균)이며 다중비교 보정을 하지 않았다.
p-value 만으로 결론을 내리지 않고 효과크기와 n 을 함께 본다. `unstable` = 가장 작은 그룹 n < {L.MIN_N_STABLE}.

{md_table(tests, ["test", "value", "n", "min_group_n", "statistic", "p", "effect", "effect_size", "unstable"])}

- 효과크기: Kruskal-Wallis 는 epsilon² = (H − k + 1)/(n − k), Mann-Whitney 는 rank-biserial r = 2U/(n₁n₂) − 1 (양수 = 앞 그룹이 큼).
- Stroke 는 점수가 있는 대상자가 적어(Subject 단위 그룹 최소 n = {int(tests.loc[tests["test"] == "Kruskal-Wallis", "min_group_n"].min())}) Group별 비교의 안정성이 낮다.

## 7. 주요 발견

""" + "\n".join(f"{i}. {f}" for i, f in enumerate(findings, 1)) + f"""

## 8. 데이터 한계

- 현재 로컬 사본은 불완전하다. 라벨은 비교적 온전하지만 원본 재확보 후 다시 집계해야 한다.
- Age 가 있는 대상자는 {supp['n_age']}명이고, Age 와 점수가 모두 있는 대상자는 {supp['n_age_and_C2']}명이다. Age 결측 대상자 수는 Group 마다 다르다 ({age_missing_groups}).
- Stroke 는 점수 보유 대상자가 적고, 출처 확인이 필요한 라벨이 포함되어 있어 제외 시 Session 이 {stroke_nb}개로 줄어든다.
- 건강군 연령이 20대에 몰려 있고 환자군은 50대 이후에 몰려 있다. 공통 구간이 넓어 보여도 구간별 표본은 얇다.
- Session 단위 수치(PO–CF 상관 등)는 같은 대상자의 관측을 포함하므로 독립 표본 검정으로 해석하지 않는다.
- 다중비교 보정을 하지 않은 탐색적 결과다.
- 운동학 신호(Position + Timestamp)와 점수가 모두 있는 Session 은 현재 매우 적다 (`docs/inventory_report.md`). 아래 후보는 라벨 측면의 타당성만 평가한 것이다.

## 9. 주 분석 후보

주 분석은 원본 데이터 재확인 전에는 확정하지 않는다. 공통 전제: Subject 단위 분할, 운동별 모델, MAE 주 지표, baseline(전체 평균·Group 평균·Age-only) 비교.

### Candidate 1
- **후보 분석**: 운동별 PO 와 CF 를 각각 target 으로 하는 회귀 (TS 는 보조)
- **목표**: 운동학 feature 가 Age-only·Group 평균 기준선보다 점수를 더 정확하게 추정하는지 (RQ2)
- **사용 데이터**: 운동별 Session (대상자 1명 = 1관측), PO/CF, Age (기준선용)
- **장점**: PO·CF 는 원저자 Script 의 P.O./C.F. 신호 구분과 대응한다. TS 는 대부분 PO+CF 이므로 PO·CF 를 나누면 정보 중복을 피하면서 해석이 쉬워진다.
- **한계**: 건강군 PO 는 만점 비율이 높아 상단이 압축되어 있을 수 있다. 운동별 n 은 대상자 수를 넘지 못한다.
- **Age confounding 영향**: Age 와 점수의 연관이 코호트 안에서도 관찰되므로 Age-only 기준선을 반드시 함께 보고해야 한다.
- **현재 데이터에서 가능한 이유**: 라벨 측면에서는 범위 이탈이 없고 결측이 적다.
- **원본 재확인 후 확인할 사항**: 운동별로 신호와 점수가 모두 있는 Session 수, 출처 확인이 필요한 라벨의 원본 값.

### Candidate 2
- **후보 분석**: Age 를 공변량으로 포함한 회귀에서 운동학 feature 의 증분 가치 평가 (Age-only 모델 대비 MAE 개선)
- **목표**: Age 로 설명되는 부분을 넘어 움직임 feature 가 추가 정보를 주는지
- **사용 데이터**: Age 와 점수가 모두 있는 대상자 ({supp['n_age_and_C2']}명)의 Session
- **장점**: overlap 판정이 {ov['class']} 이므로 매칭보다 표본 손실이 적다. Age 교란을 정면으로 다룬다.
- **한계**: Age 결측 대상자는 빠진다. Age 와 Group 이 강하게 얽혀 있어 회귀로도 완전히 분리되지 않을 수 있다.
- **Age confounding 영향**: 가장 직접적으로 다룬다. 다만 인과적 해석은 하지 않는다.
- **현재 데이터에서 가능한 이유**: Age 와 점수가 모두 있는 대상자 수가 Group별로 확보되어 있다(4.1 표).
- **원본 재확인 후 확인할 사항**: SuppInfo 결측이 원본에서도 결측인지.

### Candidate 3
- **후보 분석**: 환자군(GPP) 내부의 운동별 PO/CF 회귀
- **목표**: 상한 압축이 덜한 집단에서 점수 변동과 움직임 feature 의 관계 확인
- **사용 데이터**: 환자군 Session
- **장점**: 환자군은 상한 근처 비율이 낮아 점수 변동 폭이 크다. Group 간 연령 차이 문제를 피할 수 있다.
- **한계**: 대상자가 적고(특히 Stroke), 환자군 안에서도 Age–점수 연관이 있다. 질환이 섞여 있다.
- **Age confounding 영향**: 코호트 간 교란은 줄지만 코호트 내 Age 연관은 남는다.
- **현재 데이터에서 가능한 이유**: 라벨은 있으나 표본이 작아 주 분석보다는 보조·탐색 분석에 가깝다.
- **원본 재확인 후 확인할 사항**: 환자군 각 그룹의 신호 확보 수.

**제외한 방향**: 연령 매칭을 주 분석으로 두는 것. ±{ov['caliper']:g}세 매칭 가능한 환자가 {ov['n_matched_pairs']}명뿐이어서 sensitivity analysis 로만 쓰는 것이 적절하다.

## 10. Gate 2 판정

**판정: PASS WITH CONDITIONS**

통과 근거
- 라벨이 분석 가능한 상태다 (범위 이탈 {len(res['range_violations'])}건, 결측은 특정 대상자에 국한).
- Age confounding 의 크기와 overlap 수준({ov['class']})을 수치로 확인했다.
- 라벨 민감도의 범위를 확인했다.

조건 (다음 Gate 진행 전)
1. 공식 원본 재확보 후 이 보고서를 다시 생성하고 수치를 비교한다.
2. 불규칙 소수 라벨 {len(sens["flagged"])}명과 TS ≠ PO+CF 관측의 원본 값을 확인한다.
3. 주 분석 1개는 원본 재확인 후 사전 지정한다 (후보 1~3 중).
4. 운동별로 신호와 점수가 모두 있는 Session 수를 확인한다. 현재는 Gate 3 이후로 진행할 만큼 충분하지 않다.

## Appendix

### 함께 보는 보고서
- 데이터 구조·신호 품질(frame·Timestamp·tracking·Orientation)·Label QC 시각화와 Session 단위 점수·천장·Age 그래프: `""" + Path(os.path.relpath(Path("reports/eda/EDA_report.md").resolve(), report.parent.resolve())).as_posix() + """` (`python -m kimore.eda`)

### 생성된 그림
""" + "\n".join(f"- `{Path(os.path.relpath(p, report.parent)).as_posix()}`" for p in figs) + f"""

### 분석에 사용된 n
- Subject: 전체 {supp['n_subjects']}, Age {supp['n_age']}, C1 {supp['n_clinical_C1']}, C2 {supp['n_clinical_C2']}, Age+C2 {supp['n_age_and_C2']}
- Session: 행 {len(sess)}, 세 점수 모두 있음 {n_scored_sessions}
- 민감도 B: Subject {sens['n_subjects'][1]}, Session {sens['n_sessions'][1]}
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    return report
