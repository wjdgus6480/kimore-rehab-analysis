"""EDA·데이터 품질 테이블 (Gate 2 준비, REQUIREMENTS §4, §10.1, §10.3).

`kimore.qc` 가 만든 data/interim 산출물을 재사용하고, 그 manifest 에 없는 프레임 단위 정보
(Timestamp 간격 전체, 관절별 tracking state, 관절별 Orientation 0값)만 원본에서 **읽기 전용**으로 다시 읽는다.

사용:
    python -m kimore.eda --raw data/raw --interim data/interim --out reports/eda

산출물
- data/interim/eda/   EDA용 정리 데이터 (Session 표, 신호 QC 상세)
- reports/eda/tables/ 통계 표 (CSV)
- reports/eda/figures/ 그래프 (PNG, kimore.eda_plots)
- reports/eda/EDA_report.md  그래프별 질문과 계산된 수치

원칙
- 원본은 읽기만 한다. 값을 보정하거나 추정하지 않는다.
- 결과 수치는 코드에 하드코딩하지 않고 매번 데이터에서 계산한다.
- Timestamp 단위는 미확인이므로 원시값과 '세션 중앙값 대비 비율'로만 다룬다.
- 이 단계는 EDA 이며 ML·CV·feature 확정을 하지 않는다. 통계는 기술통계·Spearman·효과크기만 낸다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kimore import io, labels
from kimore import label_eda as L
from kimore.qc import EXERCISE_DIRS, GROUPS, RAW_KINDS

# 역할 분담 (docs/label_eda_report.md 와 함께 읽는다)
# - kimore.eda      : 데이터 구조·신호 품질·Label QC 시각화, 점수·천장·Age 의 Session 단위 탐색 그래프
# - kimore.label_eda: Gate 2 통계 판단 (Subject 단위 검정, 민감도 분석, 연령 overlap·매칭)
# 같은 계산(천장 기준·비율, TS−(PO+CF), 상관)은 label_eda 함수를 그대로 쓴다.

GROUP_ORDER: tuple[str, ...] = L.GROUP_ORDER
HEALTHY_TOP = "CG"  # 건강군 상위 폴더 (Expert, NotExpert)
COHORT_ORDER: tuple[str, ...] = L.COHORT_ORDER
CEILING_TS = labels.RANGES["TS"][1] * L.NEAR_CEILING_FRAC  # 만점의 90% = 45
GAP_FACTOR = 1.5  # kimore.qc._timestamp_stats 와 같은 '프레임 누락 의심' 기준
FORMAT_ORDER: tuple[str, ...] = ("integer", "third", "irregular")
MARKER_FILE = ".generated_by_kimore_eda"


# --------------------------------------------------------------------- loading


def load_interim(interim: str | Path) -> dict[str, pd.DataFrame]:
    """kimore.qc 산출물을 읽는다. 없으면 먼저 `python -m kimore.qc` 를 실행하라고 알린다."""
    interim = Path(interim)
    names = {
        "files": "manifest_files.csv",
        "sessions": "manifest_sessions.csv",
        "subjects": "subjects.csv",
        "long": "labels_clinical_long.csv",
        "issues": "label_qc_issues.csv",
    }
    out = {}
    for key, name in names.items():
        path = interim / name
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없습니다. 먼저 `python -m kimore.qc` 를 실행하세요.")
        out[key] = pd.read_csv(path, encoding="utf-8-sig")
    return out


def cohort_of(group_top: pd.Series) -> pd.Series:
    return np.where(group_top == HEALTHY_TOP, COHORT_ORDER[0], COHORT_ORDER[1])


# ---------------------------------------------------------------- session table


def build_eda_sessions(sessions: pd.DataFrame, subjects: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Subject × Exercise 격자 전체(manifest_sessions)에 점수·Age·파생 QC 열을 붙인다.

    점수가 없는 Session 도 남긴다(결측 시각화에 필요). 결측은 채우지 않는다.
    """
    subj = subjects[["subject_id", "group_top", "age", "gender", "has_age", "C1_has_clinical_file",
                     "C2_has_any_score", "C3_has_all_scores", "value_format"]]
    df = sessions.merge(subj, on="subject_id", how="left", validate="many_to_one")
    df = df.merge(long, on=["subject_id", "exercise"], how="left", validate="one_to_one")
    df["cohort"] = cohort_of(df["group_top"])
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)

    res = L.sum_residual(df)[["subject_id", "exercise", "ts_minus_sum", "nonzero"]]
    df = df.merge(res.rename(columns={"ts_minus_sum": "delta", "nonzero": "delta_nonzero"}),
                  on=["subject_id", "exercise"], how="left", validate="one_to_one")
    df["delta_nonzero"] = df["delta_nonzero"].fillna(False).astype(bool)
    df["ts_ge_ceiling"] = np.where(df["TS"].notna(), df["TS"] >= CEILING_TS, np.nan)
    for s in labels.SCORES:
        df[f"{s}_format"] = [labels.classify_value(v) for v in df[s]]
        lo, hi = labels.RANGES[s]
        df[f"{s}_out_of_range"] = df[s].notna() & ~df[s].between(lo, hi)
    df["_num"] = df["subject_id"].str.extract(r"(\d+)$", expand=False).astype(int)
    return df.sort_values(["group", "_num", "exercise"]).drop(columns="_num").reset_index(drop=True)


# ----------------------------------------------------------- raw signal (read-only)


def _signal_files(files: pd.DataFrame) -> pd.DataFrame:
    """kimore.qc.build_sessions 와 같은 규칙으로 '(n)' 복사본만 제외한 신호 파일."""
    is_copy = files["is_copy_name"].astype(bool) & files["duplicate_of"].notna()
    sel = files[files["kind"].isin(RAW_KINDS) & ~is_copy & files["exercise_folder"].isin(EXERCISE_DIRS)].copy()
    sel["exercise"] = sel["exercise_folder"].str.slice(2).astype(int)
    return sel


def scan_signals(raw_root: str | Path, files: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """신호 파일을 종류별로 독립적으로 읽어 프레임 단위 QC 정보를 만든다.

    한 Session 에 같은 종류 파일이 여러 개(녹화 여러 개)면 자동 선택하지 않고 skipped 로 남긴다.
    반환: ts_sessions, ts_intervals, joint_state, joint_ori_zero, skipped
    """
    raw_root = Path(raw_root)
    sel = _signal_files(files)
    ts_sess, ts_int, j_state, j_ori, skipped = [], [], [], [], []

    for (sid, ex, kind), grp in sel.groupby(["subject_folder", "exercise", "kind"], sort=True):
        key = {"subject_id": sid, "exercise": int(ex), "kind": kind}
        if len(grp) > 1:
            skipped.append({**key, "reason": f"{kind} 파일 {len(grp)}개 (자동 선택하지 않음)"})
            continue
        path = raw_root / grp.iloc[0]["path"]
        try:
            if kind == "timestamp":
                ts, info = io.read_timestamp(path)
                d = np.diff(ts.to_numpy(dtype=float))
                med = float(np.median(d)) if len(d) else math.nan
                ts_sess.append({
                    "subject_id": sid, "exercise": int(ex), "n_frames": info.n_frames,
                    "interval_median_raw": med,
                    "interval_min_raw": float(d.min()) if len(d) else math.nan,
                    "interval_max_raw": float(d.max()) if len(d) else math.nan,
                    "interval_cv": float(d.std() / d.mean()) if len(d) and d.mean() else math.nan,
                    "n_duplicate": int((d == 0).sum()),
                    "n_backward": int((d < 0).sum()),
                    "n_gap": int((d > GAP_FACTOR * med).sum()) if med > 0 else 0,
                    "max_gap_ratio": float(d.max() / med) if len(d) and med > 0 else math.nan,
                })
                ts_int.append(pd.DataFrame({
                    "subject_id": sid, "exercise": int(ex), "interval_raw": d,
                    "ratio_to_median": d / med if med > 0 else np.nan,
                }))
            elif kind == "position":
                pos, info = io.read_position(path)
                for j in io.JOINTS:
                    st = pos[f"{j}_state"].to_numpy()
                    j_state.append({
                        "subject_id": sid, "exercise": int(ex), "joint": j, "n_frames": info.n_frames,
                        "state2_ratio": float((st == 2).mean()), "state1_ratio": float((st == 1).mean()),
                        "state0_ratio": float((st == 0).mean()),
                    })
            else:
                ori, info = io.read_orientation(path)
                for j in io.JOINTS:
                    q = ori[[f"{j}_{f}" for f in io.ORIENTATION_FIELDS]].to_numpy()
                    j_ori.append({
                        "subject_id": sid, "exercise": int(ex), "joint": j, "n_frames": info.n_frames,
                        "zero_frame_ratio": float((q == 0).all(axis=1).mean()),
                        "script_no_orientation": j in io.NO_ORIENTATION_JOINTS,
                    })
        except io.ParseError as exc:
            skipped.append({**key, "reason": f"파싱 실패: {exc}"})

    return {
        "ts_sessions": pd.DataFrame(ts_sess),
        "ts_intervals": pd.concat(ts_int, ignore_index=True) if ts_int else pd.DataFrame(),
        "joint_state": pd.DataFrame(j_state),
        "joint_ori_zero": pd.DataFrame(j_ori),
        "skipped": pd.DataFrame(skipped, columns=["subject_id", "exercise", "kind", "reason"]),
    }


def attach_group(df: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.merge(subjects[["subject_id", "group", "group_top"]], on="subject_id", how="left")
    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)
    out["cohort"] = cohort_of(out["group_top"])
    return out


# ------------------------------------------------------------------- statistics


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 신뢰구간."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def spearman(x, y) -> tuple[float, int]:
    """결측 쌍을 뺀 Spearman ρ 와 n. 계산은 label_eda.correlation 과 같은 함수다."""
    c = L.correlation(x, y)
    return (c["spearman_rho"], c["n"])


def cliffs_delta(a, b) -> float:
    """Cliff's δ = P(a>b) − P(a<b). 효과크기(분포 차이)."""
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if len(a) == 0 or len(b) == 0:
        return math.nan
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def score_descriptives(es: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for s in labels.SCORES:
        g = es.groupby(by, observed=True)[s]
        t = g.agg(n="count", mean="mean", sd="std", median="median",
                  q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75), min="min", max="max")
        t["n_missing"] = g.apply(lambda v: int(v.isna().sum()))
        t.insert(0, "target", s)
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


def ceiling_table(es: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """TS ≥ 45 개수·비율(Wilson CI 추가)과 TS/PO/CF 만점 개수. 개수는 label_eda.ceiling_table 로 센다."""
    d = es[es["TS"].notna()]
    rows = []
    for key, g in d.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        c = {s: L.ceiling_table(g, s).iloc[0] for s in labels.SCORES}
        k, n = int(c["TS"]["n_near_ceiling"]), int(c["TS"]["n"])
        lo, hi = wilson_ci(k, n)
        rows.append({**dict(zip(by, key)), "n": n, "k_ts_ge_45": k, "ratio": k / n if n else math.nan,
                     "ci_low": lo, "ci_high": hi,
                     **{f"n_max_{s}": int(c[s]["n_at_max"]) for s in labels.SCORES}})
    return pd.DataFrame(rows)


def label_missing_table(es: pd.DataFrame) -> pd.DataFrame:
    """Group × Exercise × Target 결측. 분모를 두 가지로 표시한다.

    - n_subjects: 폴더 기준 전체 대상자
    - n_c1: ClinicalAssessment 파일이 있는 대상자 (값 결측은 이 안에서 센다)
    """
    rows = []
    for (grp, ex), g in es.groupby(["group", "exercise"], observed=True):
        c1 = g[g["C1_has_clinical_file"].astype(bool)]
        for s in labels.SCORES:
            rows.append({"group": grp, "exercise": int(ex), "target": s, "n_subjects": len(g),
                         "n_no_clinical_file": int((~g["C1_has_clinical_file"].astype(bool)).sum()),
                         "n_c1": len(c1), "n_missing_in_c1": int(c1[s].isna().sum()),
                         "n_missing_total": int(g[s].isna().sum())})
    return pd.DataFrame(rows)


def value_format_tables(es: pd.DataFrame, subjects: pd.DataFrame) -> dict[str, pd.DataFrame]:
    long = es.melt(id_vars=["subject_id", "group", "exercise"], value_vars=list(labels.SCORES),
                   var_name="target", value_name="value").dropna(subset=["value"])
    long["format"] = [labels.classify_value(v) for v in long["value"]]
    long["frac_part"] = long["value"] - np.floor(long["value"])
    per_value = long.groupby(["group", "target", "format"], observed=True).size().rename("n").reset_index()

    c2 = subjects[subjects["C2_has_any_score"].astype(bool)]
    per_subject = c2.groupby(["group", "value_format"]).size().rename("n").reset_index()

    irr = long[long["format"] == "irregular"]
    irregular = (irr.groupby("subject_id")
                 .agg(group=("group", "first"), n_irregular_values=("value", "size"),
                      examples=("value", lambda v: ", ".join(f"{x:.4f}" for x in list(v)[:4])))
                 .reset_index())
    return {"values": long, "per_value": per_value, "per_subject": per_subject, "irregular": irregular}


def age_tables(subjects: pd.DataFrame, es: pd.DataFrame) -> dict[str, pd.DataFrame]:
    s = subjects.copy()
    s["cohort"] = cohort_of(s["group_top"])
    by_group = (s.groupby("group").agg(n_subjects=("subject_id", "size"), n_age=("has_age", "sum"))
                .reindex(GROUP_ORDER))
    a = s[s["has_age"].astype(bool)]
    stats = a.groupby("group")["age"].agg(["mean", "std", "median", "min", "max"]).reindex(GROUP_ORDER)
    by_group = by_group.join(stats)
    by_group["n_age_missing"] = by_group["n_subjects"] - by_group["n_age"]

    h, p = (a.loc[a["cohort"] == c, "age"] for c in COHORT_ORDER)
    cohort = pd.DataFrame([{
        "n_healthy": len(h), "n_patient": len(p),
        "median_healthy": h.median(), "median_patient": p.median(),
        "cliffs_delta_patient_vs_healthy": cliffs_delta(p, h),
    }])

    rows = []
    for ex in labels.EXERCISES:
        e = es[es["exercise"] == ex]
        for s_ in labels.SCORES:
            for scope, sub in (("전체", e), *((c, e[e["cohort"] == c]) for c in COHORT_ORDER)):
                rho, n = spearman(sub["age"], sub[s_])
                rows.append({"exercise": ex, "target": s_, "scope": scope, "spearman_rho": rho, "n": n})
    corr = pd.DataFrame(rows)
    return {"by_group": by_group.reset_index(), "cohort": cohort, "age_score_spearman": corr}


# ----------------------------------------------------------------------- output


def _prepare_out(path: Path) -> None:
    """생성 전용 폴더만 쓴다. 표식 없는 기존 폴더에 파일이 있으면 덮어쓰지 않고 중단한다."""
    if path.exists() and any(path.iterdir()) and not (path / MARKER_FILE).exists():
        raise FileExistsError(f"{path} 에 kimore.eda 가 만들지 않은 파일이 있습니다. 다른 --out 을 지정하세요.")
    path.mkdir(parents=True, exist_ok=True)
    (path / MARKER_FILE).write_text("kimore.eda 가 생성·갱신하는 폴더입니다.\n", encoding="utf-8")


def _csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_all(raw_root: str | Path, interim: str | Path) -> dict:
    d = load_interim(interim)
    subjects = d["subjects"]
    es = build_eda_sessions(d["sessions"], subjects, d["long"])
    sig = scan_signals(raw_root, d["files"])
    for k in ("ts_sessions", "ts_intervals", "joint_state", "joint_ori_zero"):
        sig[k] = attach_group(sig[k], subjects)
    scored = es[es["C1_has_clinical_file"].astype(bool)]
    tables = {
        "score_desc_by_group": score_descriptives(scored, ["group"]),
        "score_desc_by_exercise": score_descriptives(scored, ["exercise"]),
        "score_desc_by_group_exercise": score_descriptives(scored, ["group", "exercise"]),
        "ceiling_by_group": ceiling_table(es, ["group"]),
        "ceiling_by_cohort": ceiling_table(es, ["cohort"]),
        "ceiling_by_group_exercise": ceiling_table(es, ["group", "exercise"]),
        "ceiling_by_cohort_exercise": ceiling_table(es, ["cohort", "exercise"]),
        "label_missing": label_missing_table(es),
        "out_of_range": es[es[[f"{s}_out_of_range" for s in labels.SCORES]].any(axis=1)][
            ["subject_id", "group", "exercise", *labels.SCORES]],
        "delta_nonzero": es[es["delta_nonzero"]][["subject_id", "group", "exercise", "TS", "PO", "CF", "delta"]],
    }
    vf = value_format_tables(es, subjects)
    tables.update({"format_per_value": vf["per_value"], "format_per_subject": vf["per_subject"],
                   "format_irregular_subjects": vf["irregular"]})
    ag = age_tables(subjects, es)
    tables.update({"age_by_group": ag["by_group"], "age_cohort": ag["cohort"],
                   "age_score_spearman": ag["age_score_spearman"]})
    return {"interim": d, "es": es, "signal": sig, "tables": tables, "format_values": vf["values"]}


def run(raw_root, interim, out, eda_data=None, make_figures: bool = True) -> dict:
    out = Path(out)
    eda_data = Path(eda_data) if eda_data else Path(interim) / "eda"
    for p in (eda_data, out):
        _prepare_out(p)
    (out / "tables").mkdir(exist_ok=True)

    res = build_all(raw_root, interim)
    _csv(res["es"], eda_data / "eda_sessions.csv")
    for k in ("ts_sessions", "ts_intervals", "joint_state", "joint_ori_zero", "skipped"):
        _csv(res["signal"][k], eda_data / f"signal_{k}.csv")
    for name, t in res["tables"].items():
        _csv(t, out / "tables" / f"{name}.csv")

    if make_figures:
        from kimore import eda_plots  # matplotlib 은 그림을 만들 때만 필요

        figs = eda_plots.make_all(res, out / "figures")
        eda_plots.write_report(res, figs, out / "EDA_report.md")
        res["figures"] = figs
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="KIMORE EDA·데이터 품질 시각화 (원본은 읽기만 함)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--interim", default="data/interim", help="kimore.qc 산출물 폴더")
    ap.add_argument("--eda-data", default=None, help="EDA 정리 데이터 폴더 (기본: <interim>/eda)")
    ap.add_argument("--out", default="reports/eda", help="그래프·통계표·보고서 폴더")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    res = run(args.raw, args.interim, args.out, args.eda_data, make_figures=not args.no_figures)
    es = res["es"]
    print(f"sessions={len(es)} scored_sessions={int(es['TS'].notna().sum())} "
          f"ts_files={len(res['signal']['ts_sessions'])} skipped={len(res['signal']['skipped'])} "
          f"figures={len(res.get('figures', []))}")


if __name__ == "__main__":
    main()
