"""Gate 2 라벨·Age EDA 그림 (kimore.label_eda 분석 결과를 그린다).

분석 로직(kimore.label_eda)과 시각화를 분리한다. 분석 모듈은 matplotlib 없이 import 되며
DataFrame / dict 를 반환하므로, 이후 API 계층은 분석 모듈만 호출하면 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kimore import labels
from kimore.label_eda import COHORT_ORDER, GROUP_ORDER, SCORES, correlation, subject_scores  # noqa: F401


# -------------------------------------------------------------------- plots


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if fam in names:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    return plt


_COLORS = {"Expert": "#4C72B0", "NotExpert": "#55A868", "BackPain": "#C44E52", "Parkinson": "#8172B2",
           "Stroke": "#CCB974", "건강군": "#4C72B0", "환자군": "#C44E52"}


def _box_strip(ax, values: list[np.ndarray], names: list[str], positions=None, width=0.6):
    rng = np.random.default_rng(0)
    positions = positions if positions is not None else np.arange(1, len(values) + 1)
    data = [v if len(v) else np.array([np.nan]) for v in values]
    ax.boxplot(data, positions=positions, widths=width, showfliers=False,
               medianprops={"color": "black"}, boxprops={"color": "#555"})
    for pos, v, name in zip(positions, values, names):
        if len(v):
            ax.scatter(pos + rng.uniform(-width / 4, width / 4, len(v)), v, s=12, alpha=0.7,
                       color=_COLORS.get(name, "#333"), zorder=3)
    return positions


def _annotate_n(ax, positions, values):
    top = ax.get_ylim()[1]
    for pos, v in zip(positions, values):
        ax.text(pos, top, f"n={len(v)}", ha="center", va="bottom", fontsize=8)


def plot_score_by_group(sessions: pd.DataFrame, score: str, path: Path) -> Path:
    plt = _setup_mpl()
    groups = [g for g in GROUP_ORDER if g in set(sessions["group"])]
    vals = [sessions.loc[sessions["group"] == g, score].dropna().to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = _box_strip(ax, vals, groups)
    ax.set_xticks(pos, groups)
    ax.set_ylim(*labels.RANGES[score])
    ax.set_ylabel(score)
    ax.set_title(f"Group별 {score} (Session 단위: 대상자×운동)", pad=16)
    _annotate_n(ax, pos, vals)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_score_by_group_exercise(sessions: pd.DataFrame, score: str, path: Path) -> Path:
    plt = _setup_mpl()
    groups = [g for g in GROUP_ORDER if g in set(sessions["group"])]
    fig, axes = plt.subplots(1, len(labels.EXERCISES), figsize=(16, 4), sharey=True)
    for ax, ex in zip(axes, labels.EXERCISES):
        e = sessions[sessions["exercise"] == ex]
        vals = [e.loc[e["group"] == g, score].dropna().to_numpy() for g in groups]
        pos = _box_strip(ax, vals, groups)
        ax.set_xticks(pos, [g[:4] for g in groups], fontsize=8)
        ax.set_ylim(*labels.RANGES[score])
        ax.set_title(f"Ex{ex}", pad=14)
        _annotate_n(ax, pos, vals)
    axes[0].set_ylabel(score)
    fig.suptitle(f"Group × Exercise별 {score} (Session 단위)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_po_cf(sessions: pd.DataFrame, path: Path) -> Path:
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for g in GROUP_ORDER:
        sub = sessions[sessions["group"] == g].dropna(subset=["PO", "CF"])
        ax.scatter(sub["PO"], sub["CF"], s=14, alpha=0.7, color=_COLORS[g], label=f"{g} (n={len(sub)})")
    ax.set_xlim(*labels.RANGES["PO"])
    ax.set_ylim(*labels.RANGES["CF"])
    ax.set_xlabel("PO")
    ax.set_ylabel("CF")
    ax.set_title("PO vs CF (Session 단위)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_age_by_group(subjects: pd.DataFrame, path: Path) -> Path:
    plt = _setup_mpl()
    s = subjects[subjects["has_age"]]
    groups = [g for g in GROUP_ORDER if g in set(s["group"])]
    vals = [s.loc[s["group"] == g, "age"].to_numpy(dtype=float) for g in groups]
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = _box_strip(ax, vals, groups)
    ax.set_xticks(pos, groups)
    ax.set_ylabel("Age")
    ax.set_title("Group별 Age (Subject 단위, Age 있는 대상자만)", pad=16)
    _annotate_n(ax, pos, vals)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_age_overlap(subjects: pd.DataFrame, overlap: dict, path: Path) -> Path:
    plt = _setup_mpl()
    s = subjects[subjects["has_age"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(10, 95, 5)
    for c in COHORT_ORDER:
        a = s.loc[s["cohort"] == c, "age"].to_numpy(dtype=float)
        ax.hist(a, bins=bins, alpha=0.55, color=_COLORS[c], label=f"{c} (n={len(a)})")
    if not np.isnan(overlap.get("common_low", np.nan)):
        ax.axvspan(overlap["common_low"], overlap["common_high"], color="grey", alpha=0.12, label="공통 연령 구간")
    ax.set_xlabel("Age")
    ax.set_ylabel("대상자 수")
    ax.set_title("건강군·환자군 연령 분포 (5세 구간)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_age_vs_scores(sessions: pd.DataFrame, path: Path) -> Path:
    plt = _setup_mpl()
    subj = subject_scores(sessions).dropna(subset=["age"])
    fig, axes = plt.subplots(1, len(SCORES), figsize=(14, 4))
    for ax, s in zip(axes, SCORES):
        for c in COHORT_ORDER:
            sub = subj[subj["cohort"] == c]
            ax.scatter(sub["age"], sub[s], s=16, alpha=0.75, color=_COLORS[c], label=c)
        c = correlation(subj["age"], subj[s])
        ax.set_title(f"Age vs {s}: ρ={c['spearman_rho']:.2f}, n={c['n']}")
        ax.set_xlabel("Age")
        ax.set_ylabel(f"{s} (대상자 운동 평균)")
        ax.set_ylim(*labels.RANGES[s])
    axes[0].legend(fontsize=8)
    fig.suptitle("Age vs 점수 (Subject 단위)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sensitivity(sessions: pd.DataFrame, flagged: list[str], path: Path) -> Path:
    plt = _setup_mpl()
    groups = [g for g in GROUP_ORDER if g in set(sessions["group"])]
    b = sessions[~sessions["subject_id"].isin(flagged)]
    fig, axes = plt.subplots(1, len(SCORES), figsize=(16, 4))
    for ax, s in zip(axes, SCORES):
        va = [sessions.loc[sessions["group"] == g, s].dropna().to_numpy() for g in groups]
        vb = [b.loc[b["group"] == g, s].dropna().to_numpy() for g in groups]
        pa = np.arange(len(groups)) * 2.5 + 1
        pb = pa + 0.9
        ax.boxplot([v if len(v) else [np.nan] for v in va], positions=pa, widths=0.7, showfliers=False,
                   patch_artist=True, boxprops={"facecolor": "#ddd"}, medianprops={"color": "black"})
        ax.boxplot([v if len(v) else [np.nan] for v in vb], positions=pb, widths=0.7, showfliers=False,
                   patch_artist=True, boxprops={"facecolor": "#9ecae1"}, medianprops={"color": "black"})
        ax.set_xticks(pa + 0.45, [f"{g[:4]}\nA{len(x)}/B{len(y)}" for g, x, y in zip(groups, va, vb)], fontsize=8)
        ax.set_ylim(*labels.RANGES[s])
        ax.set_title(s)
    axes[0].set_ylabel("점수 (Session 단위)")
    fig.suptitle("라벨 민감도: A=불규칙 소수 포함(회색), B=제외(파랑), 아래 숫자는 Session n")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path



def make_figures(res: dict, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    s, subj = res["sessions"], res["subjects"]
    figs = []
    for sc in SCORES:
        figs.append(plot_score_by_group(s, sc, fig_dir / f"group_{sc}.png"))
    for sc in SCORES:
        figs.append(plot_score_by_group_exercise(s, sc, fig_dir / f"group_exercise_{sc}.png"))
    figs.append(plot_po_cf(s, fig_dir / "po_vs_cf.png"))
    figs.append(plot_age_by_group(subj, fig_dir / "age_by_group.png"))
    figs.append(plot_age_overlap(subj, res["overlap"], fig_dir / "age_overlap.png"))
    figs.append(plot_age_vs_scores(s, fig_dir / "age_vs_scores.png"))
    figs.append(plot_sensitivity(s, res["flagged"], fig_dir / "label_sensitivity.png"))
    return figs
