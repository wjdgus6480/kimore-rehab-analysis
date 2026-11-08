"""EDA·데이터 품질 그래프와 보고서 (kimore.eda 결과를 그린다).

각 그래프는 하나의 질문에 답한다. 질문은 FIGURES 목록에 함께 적고 보고서에 그대로 옮긴다.
모든 수치(표본 수 n, 비율 등)는 그릴 때 데이터에서 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from kimore import io, labels  # noqa: E402
from kimore.eda import CEILING_TS, COHORT_ORDER, FORMAT_ORDER, GAP_FACTOR, GROUP_ORDER, wilson_ci  # noqa: E402

# 참조 팔레트(고정 순서 categorical) — 색은 Group 정체성만 나타내고 순위와 무관하다.
GROUP_COLORS = dict(zip(GROUP_ORDER, ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#e87ba4"]))
COHORT_COLORS = {COHORT_ORDER[0]: "#2a78d6", COHORT_ORDER[1]: "#eb6834"}
FORMAT_COLORS = {"integer": "#2a78d6", "third": "#1baf7a", "irregular": "#e34948"}
FORMAT_KO = {"integer": "정수", "third": "1/3 단위", "irregular": "불규칙 소수"}
INK, INK2, GRID, ABSENT, ALERT = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7", "#e34948"
SEQ = "Blues"
SEQ_WARN = "Oranges"
PRELIM = "개발용·예비 결과 (현재 로컬 사본)"

_INSTALLED = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_FONTS = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in _INSTALLED] + ["DejaVu Sans"]

plt.rcParams.update({
    "font.family": _FONTS,
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",  # log 축 지수 표기의 '−' 글리프
    "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9.5,
    "legend.frameon": False,
})


@dataclass
class Fig:
    file: str
    section: str
    title: str
    question: str
    finding: str = ""


# ------------------------------------------------------------------ helpers


def _save(fig, out: Path, name: str, title: str) -> str:
    fig.suptitle(title, x=0.01, y=1.0, ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.99, 1.0, PRELIM, ha="right", va="bottom", fontsize=8, color=INK2)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


_PLACEHOLDER_INFO = {
    "B1": ("B1_frames_per_session", "B1. Session별 frame 수 분포", "frame 수 분포", "Session별 frame 수는 어떤 분포인가?"),
    "B2": ("B2_timestamp_intervals", "B2. Timestamp 간격은 일정한가?", "Timestamp 간격", "Timestamp 간격은 일정한가?"),
    "B3": ("B3_frame_drops", "B3. frame drop(누락 의심)", "frame drop", "frame drop이 있는가?"),
    "B4": ("B4_inferred_tracking", "B4. inferred tracking 비율", "inferred tracking 비율", "inferred tracking 비율은 어느 정도인가?"),
    "B5": ("B5_orientation_zero_joints", "B5. Orientation 구조적 0값", "Orientation 구조적 0값",
           "Orientation의 구조적 0값은 어떤 관절에 존재하는가?"),
}


def _placeholder(out: Path, tag: str, reason: str) -> Fig:
    """그릴 데이터가 없을 때도 그래프 자리를 남겨 '없음'을 눈으로 확인하게 한다."""
    name, title, short, question = _PLACEHOLDER_INFO[tag]
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    ax.text(0.5, 0.5, reason, ha="center", va="center", fontsize=12, color=INK2)
    return Fig(_save(fig, out, name, title), "B", short, question, reason)


def _n(v) -> int:
    """표시용 개수. 해당 범주가 아예 없어 NaN 이면 0."""
    return 0 if v is None or v != v else int(v)


def _jitter(n: int, seed: int, width: float = 0.18) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def _box_strip(ax, data: pd.DataFrame, x: str, y: str, order, color_by: str | None = None,
               colors: dict | None = None, ref_lines: tuple = (), show_n: bool = True):
    """boxplot + jitter. x 범주 아래에 n 을 표기한다."""
    vals = [data.loc[data[x] == k, y].dropna().to_numpy() for k in order]
    pos = np.arange(len(order))
    nonempty = [i for i, v in enumerate(vals) if len(v)]
    if nonempty:
        ax.boxplot([vals[i] for i in nonempty], positions=pos[nonempty], widths=0.5, showfliers=False,
                   patch_artist=True, medianprops={"color": INK, "linewidth": 1.6},
                   boxprops={"facecolor": "none", "edgecolor": INK2},
                   whiskerprops={"color": INK2}, capprops={"color": INK2})
    for i, k in enumerate(order):
        sub = data[(data[x] == k) & data[y].notna()]
        if sub.empty:
            continue
        if color_by:
            c = [colors[v] for v in sub[color_by]]
        else:
            c = colors.get(k, "#2a78d6") if colors else "#2a78d6"
        ax.scatter(i + _jitter(len(sub), seed=i), sub[y], s=16, c=c, alpha=0.75,
                   edgecolors="white", linewidths=0.5, zorder=3)
    for v, lab in ref_lines:
        ax.axhline(v, color=INK2, ls="--", lw=1)
        ax.text(len(order) - 0.5, v, f" {lab}", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{k}\nn={len(v)}" if show_n else str(k) for k, v in zip(order, vals)])
    ax.grid(axis="x", visible=False)


def _annot_heatmap(ax, mat: np.ndarray, text: np.ndarray, rows, cols, cmap=SEQ, vmin=0, vmax=None,
                   cbar_label: str = "", fig=None):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    vmax_eff = vmax if vmax is not None else np.nanmax(mat) if np.isfinite(mat).any() else 1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            dark = np.isfinite(v) and v > (vmin + 0.6 * (vmax_eff - vmin))
            ax.text(j, i, text[i, j], ha="center", va="center", fontsize=8,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(cbar_label, color=INK2)
        cb.outline.set_visible(False)
    return im


def _bar_labels(ax, bars, fmt="{:.0f}"):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h), ha="center", va="bottom",
                    fontsize=8, color=INK)


def _group_legend(ax, groups=GROUP_ORDER, loc="upper right", **kw):
    ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=g) for g in groups], loc=loc, fontsize=8, **kw)


EX_LABELS = [f"Ex{e}" for e in labels.EXERCISES]


# ------------------------------------------------------------- A. overview


def fig_overview(res, out) -> list[Fig]:
    subjects = res["interim"]["subjects"]
    es = res["es"]
    figs = []

    # A1 Group별 Subject 수 + 라벨 보유
    cols = [("n", "폴더 기준 전체"), ("has_suppinfo", "SuppInfo 있음"), ("has_age", "Age 있음"),
            ("C1_has_clinical_file", "C1 Clinical 파일"), ("C2_has_any_score", "C2 점수 ≥1"),
            ("C3_has_all_scores", "C3 점수 15개")]
    t = subjects.assign(n=1).groupby("group")[[c for c, _ in cols]].sum().reindex(GROUP_ORDER)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.13
    shades = ["#0b3d7a", "#2a78d6", "#7fb0ea", "#1baf7a", "#5fd0a6", "#a9e6cf"]
    for i, ((c, lab), col) in enumerate(zip(cols, shades)):
        bars = ax.bar(np.arange(len(t)) + (i - 2.5) * w, t[c], w * 0.92, label=lab, color=col)
        _bar_labels(ax, bars)
    ax.set_xticks(range(len(t)), t.index)
    ax.set_ylabel("Subject 수")
    ax.legend(ncol=3, loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    total = int(t["n"].sum())
    finding = (f"전체 {total}명 · SuppInfo {_n(t['has_suppinfo'].sum())}명 · Age {_n(t['has_age'].sum())}명 · "
               f"C1 {_n(t['C1_has_clinical_file'].sum())} / C2 {_n(t['C2_has_any_score'].sum())} / "
               f"C3 {_n(t['C3_has_all_scores'].sum())}명")
    figs.append(Fig(_save(fig, out, "A1_subjects_by_group", "A1. Group별 Subject 수와 라벨 보유"), "A",
                    "Group별 Subject 수", "전체 Subject 수, Group별 수, Clinical·SuppInfo 보유 Subject 수는?", finding))

    # A2 Exercise별 Session 수 (가용 단계별)
    tiers = [("session_folder_exists", "운동 폴더 있음"), ("has_position", "Position"),
             ("has_timestamp", "Timestamp"), ("analyzable_signal", "신호 분석 가능(QC)"),
             ("analyzable_with_scores", "신호 + 점수")]
    t2 = es.groupby("exercise")[[c for c, _ in tiers]].sum()
    n_grid = es.groupby("exercise").size()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.16
    shades = ["#0b3d7a", "#2a78d6", "#7fb0ea", "#1baf7a", "#eb6834"]
    for i, ((c, lab), col) in enumerate(zip(tiers, shades)):
        bars = ax.bar(np.arange(5) + (i - 2) * w, t2[c], w * 0.92, label=lab, color=col)
        _bar_labels(ax, bars)
    ax.set_xticks(range(5), [f"Ex{e}\n(격자 {n_grid[e]})" for e in t2.index])
    ax.set_ylabel("Session 수")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, n_grid.max() * 1.25)
    finding = ", ".join(f"Ex{e} {_n(t2.loc[e, 'analyzable_with_scores'])}" for e in t2.index)
    figs.append(Fig(_save(fig, out, "A2_sessions_by_exercise", "A2. Exercise별 Session 수 (가용 단계별)"), "A",
                    "Exercise별 Session 수", "Exercise별로 실제 분석 가능한 Session은 몇 개인가?",
                    f"신호+점수 Session: {finding}"))

    # A3 Group × Exercise (heatmap 4 panel)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, (c, lab) in zip(axes, [("session_folder_exists", "운동 폴더 있음"),
                                   ("pos_ts", "Position + Timestamp"),
                                   ("pos_ts_ori", "Position + Timestamp + Orientation"),
                                   ("analyzable_with_scores", "신호 QC 통과 + 점수")]):
        d = es.assign(pos_ts=es["has_position"] & es["has_timestamp"],
                      pos_ts_ori=es["has_position"] & es["has_timestamp"] & es["has_orientation"])
        m = d.pivot_table(index="group", columns="exercise", values=c, aggfunc="sum", observed=False)
        n = d.pivot_table(index="group", columns="exercise", values=c, aggfunc="size", observed=False)
        m, n = m.reindex(GROUP_ORDER), n.reindex(GROUP_ORDER)
        txt = np.array([[f"{_n(a)}/{_n(b)}" for a, b in zip(r1, r2)] for r1, r2 in zip(m.values, n.values)])
        _annot_heatmap(ax, (m / n).values.astype(float), txt, GROUP_ORDER, EX_LABELS, vmax=1)
        ax.set_title(lab, fontsize=10)
    figs.append(Fig(_save(fig, out, "A3_sessions_group_exercise", "A3. Group × Exercise Session 수 (칸: 개수/격자, 색: 비율)"),
                    "A", "Group × Exercise Session 수", "어느 Group·Exercise 조합에 분석 가능한 Session이 있는가?"))

    # A4 파일 종류별 존재율
    kinds = [("has_position", "Position"), ("has_timestamp", "Timestamp"), ("has_orientation", "Orientation"),
             ("has_clinical", "Clinical 파일"), ("has_scores", "해당 운동 점수"), ("has_suppinfo", "SuppInfo")]
    m = es.groupby("group", observed=False)[[c for c, _ in kinds]].mean().reindex(GROUP_ORDER)
    k = es.groupby("group", observed=False)[[c for c, _ in kinds]].sum().reindex(GROUP_ORDER)
    n = es.groupby("group", observed=False).size().reindex(GROUP_ORDER)
    txt = np.array([[f"{_n(v)}/{_n(n[g])}" for v in k.loc[g]] for g in GROUP_ORDER])
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    _annot_heatmap(ax, m.values.astype(float), txt, GROUP_ORDER, [lab for _, lab in kinds], vmax=1,
                   cbar_label="존재 비율 (Session 기준)", fig=fig)
    figs.append(Fig(_save(fig, out, "A4_file_presence", "A4. 파일 종류별 존재 여부 (Group, Session 기준)"), "A",
                    "파일 종류별 존재 여부", "각 Session에 Position / Orientation / Timestamp / 라벨이 존재하는가?"))

    # A5 Session completeness (Subject × [Exercise × 파일])
    parts = [("has_position", "P"), ("has_timestamp", "T"), ("has_orientation", "O"), ("has_scores", "S")]
    subj_order = es.drop_duplicates("subject_id")[["subject_id", "group"]]
    rows = list(subj_order["subject_id"])
    piv = {c: es.pivot(index="subject_id", columns="exercise", values=c).reindex(rows) for c, _ in parts}
    col_labels, mat = [], []
    for e in labels.EXERCISES:
        for c, ab in parts:
            col_labels.append(f"Ex{e}\n{ab}")
            mat.append(piv[c][e].astype(float).values)
    supp = es.drop_duplicates("subject_id").set_index("subject_id").reindex(rows)
    col_labels += ["Supp\nInfo", "Age"]
    mat += [supp["has_suppinfo"].astype(float).values, supp["has_age"].astype(bool).astype(float).values]
    mat = np.column_stack(mat)
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.imshow(mat, cmap=ListedColormap([ABSENT, "#2a78d6"]), vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), rows, fontsize=6.5)
    ax.grid(False)
    for x in range(4, len(col_labels) - 2, 4):
        ax.axvline(x - 0.5, color="white", lw=2.5)
    ax.axvline(len(col_labels) - 2.5, color="white", lw=2.5)
    grp = list(subj_order["group"])
    for i in range(1, len(grp)):
        if grp[i] != grp[i - 1]:
            ax.axhline(i - 0.5, color=INK, lw=1.2)
    starts = [0] + [i for i in range(1, len(grp)) if grp[i] != grp[i - 1]] + [len(grp)]
    for a, b in zip(starts[:-1], starts[1:]):
        ax.text(len(col_labels) + 0.3, (a + b - 1) / 2, grp[a], va="center", fontsize=9, fontweight="bold")
    ax.legend(handles=[Patch(color="#2a78d6", label="있음 ✓"), Patch(color=ABSENT, label="없음 ✗")],
              loc="upper left", bbox_to_anchor=(1.0, 1.03), fontsize=8)
    ax.set_xlabel("P=Position, T=Timestamp, O=Orientation, S=해당 운동 TS·PO·CF 모두 있음", color=INK2)
    figs.append(Fig(_save(fig, out, "A5_session_completeness", "A5. Session completeness (Subject × Exercise × 파일)"),
                    "A", "Session completeness", "어떤 Subject·Exercise에서 어떤 파일이 빠졌는가? (표 형태: data/interim/eda/eda_sessions.csv)"))
    return figs


# ---------------------------------------------------------- B. signal quality


def fig_quality(res, out) -> list[Fig]:
    sig, es = res["signal"], res["es"]
    ts, ints, js, jo = sig["ts_sessions"], sig["ts_intervals"], sig["joint_state"], sig["joint_ori_zero"]
    figs = []

    # B1 frame 수 분포
    def _b1() -> Fig:
        cols = ["subject_id", "exercise", "group", "n_frames"]
        pos_n = js.drop_duplicates(["subject_id", "exercise"])[cols] if not js.empty else pd.DataFrame(columns=cols)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
        for ax, d, lab in ((axes[0], ts, "Timestamp 파일"), (axes[1], pos_n, "Position 파일")):
            _box_strip(ax, d, "exercise", "n_frames", list(labels.EXERCISES), color_by="group", colors=GROUP_COLORS)
            ax.set_xticklabels([f"Ex{e}\nn={_n((d['exercise'] == e).sum())}" for e in labels.EXERCISES])
            ax.set_title(f"{lab} 기준 (Session {len(d)}개)", fontsize=10)
            ax.set_ylabel("frame 수")
        _group_legend(axes[1])
        q = ts["n_frames"].quantile([0, 0.5, 1]).astype(int).tolist() if len(ts) else [0, 0, 0]
        return (Fig(_save(fig, out, "B1_frames_per_session", "B1. Session별 frame 수 분포"), "B",
                        "frame 수 분포", "Session별 frame 수는 어떤 분포인가?",
                        f"Timestamp 기준 최소 {q[0]} · 중앙값 {q[1]} · 최대 {q[2]} frame"))

    # B2 Timestamp 간격
    def _b2() -> Fig:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
        r = ints["ratio_to_median"].dropna()
        cap = 6.0
        axes[0].hist(r.clip(upper=cap), bins=np.arange(0.5, cap + 0.1, 0.1), color="#2a78d6", edgecolor="none")
        axes[0].set_yscale("log")
        axes[0].axvline(GAP_FACTOR, color=ALERT, ls="--", lw=1.2)
        axes[0].text(GAP_FACTOR, axes[0].get_ylim()[1], f" 누락 의심 기준 {GAP_FACTOR}×", color=ALERT,
                     va="top", fontsize=8)
        n_cap = int((r > cap).sum())
        axes[0].text(0.98, 0.95, f"{cap:g}× 이상 {n_cap}개 (최대 {r.max():.1f}×)", transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=8, color=INK2)
        axes[0].set_xlabel(f"frame 간격 / 해당 Session 중앙값 ({cap:g} 이상은 마지막 칸에 합침)")
        axes[0].set_ylabel("간격 수 (log)")
        n_gap = int((r > GAP_FACTOR).sum())
        axes[0].set_title(f"전체 간격 {len(r):,}개 중 {GAP_FACTOR}× 초과 {n_gap}개 ({n_gap / max(len(r), 1):.2%})", fontsize=10)
        axes[1].scatter(ts["interval_median_raw"], ts["interval_cv"], s=18,
                        c=[GROUP_COLORS[g] for g in ts["group"]], edgecolors="white", linewidths=0.5)
        axes[1].set_xlabel("Session 중앙 간격 (원시 단위, 단위 미확인)")
        axes[1].set_ylabel("간격 변동계수 CV")
        axes[1].set_title(f"Session별 중앙 간격·변동 (n={len(ts)})", fontsize=10)
        axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
        _group_legend(axes[1])
        med = ts["interval_median_raw"]
        return (Fig(_save(fig, out, "B2_timestamp_intervals", "B2. Timestamp 간격은 일정한가?"), "B",
                        "Timestamp 간격", "Timestamp 간격은 일정한가?",
                        f"Session 중앙 간격 {med.min():,.0f} ~ {med.max():,.0f} (원시 단위); "
                        f"중복 {_n(ts['n_duplicate'].sum())} · 역행 {_n(ts['n_backward'].sum())}건"))

    # B3 frame drop
    def _b3() -> Fig:
        d = ts.assign(gap_ratio=ts["n_gap"] / (ts["n_frames"] - 1).clip(lower=1))
        d = d.sort_values(["group", "n_gap"], ascending=[True, False])
        fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
        x = np.arange(len(d))
        cols = [GROUP_COLORS[g] for g in d["group"]]
        axes[0].bar(x, d["n_gap"], color=cols, width=0.85)
        axes[0].set_ylabel(f"간격 > {GAP_FACTOR}× 중앙값 (개)")
        axes[1].bar(x, d["max_gap_ratio"], color=cols, width=0.85)
        axes[1].set_yscale("log")
        axes[1].axhline(GAP_FACTOR, color=ALERT, ls="--", lw=1)
        axes[1].set_ylabel("최대 간격 / 중앙값 (log)")
        axes[1].set_xticks(x, [f"{s} Ex{e}" for s, e in zip(d["subject_id"], d["exercise"])], rotation=90, fontsize=5.5)
        for ax in axes:
            ax.grid(axis="x", visible=False)
        _group_legend(axes[0], ncol=5)
        n_sess = int((d["n_gap"] > 0).sum())
        return (Fig(_save(fig, out, "B3_frame_drops", "B3. frame drop(누락 의심) — Session별"), "B",
                        "frame drop", "frame drop이 있는가? 어느 Session에 몰려 있는가?",
                        f"누락 의심 간격이 1개 이상인 Session {n_sess}/{len(d)}"))

    # B4 inferred tracking 비율
    def _b4() -> Fig:
        sess = js.groupby(["subject_id", "exercise", "group"], observed=True)["state1_ratio"].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.4]})
        _box_strip(axes[0], sess, "group", "state1_ratio", GROUP_ORDER, colors=GROUP_COLORS)
        axes[0].set_ylabel("state=1 (inferred 추정) 비율, Session 평균")
        axes[0].set_title(f"Session별 (Position 파일 {len(sess)}개)", fontsize=10)
        m = js.pivot_table(index="joint", columns="group", values="state1_ratio", aggfunc="mean", observed=False)
        m = m.reindex(index=list(io.JOINTS), columns=GROUP_ORDER)
        ns = sess.groupby("group", observed=False).size().reindex(GROUP_ORDER)
        txt = np.array([[("" if np.isnan(v) else f"{v:.0%}") for v in r_] for r_ in m.values])
        _annot_heatmap(axes[1], m.values.astype(float), txt, list(io.JOINTS),
                       [f"{g}\n(n={_n(ns[g])})" for g in GROUP_ORDER], cmap=SEQ_WARN, cbar_label="state=1 비율", fig=fig)
        axes[1].set_title("관절 × Group 평균", fontsize=10)
        overall = float(js["state1_ratio"].mean()) if len(js) else np.nan
        top = js.groupby("joint")["state1_ratio"].mean().sort_values(ascending=False).head(3)
        return (Fig(_save(fig, out, "B4_inferred_tracking", "B4. inferred tracking(state=1) 비율 — QC 전용, feature 아님"), "B",
                        "inferred tracking 비율", "inferred tracking 비율은 어느 정도인가? 어느 관절에서 높은가?",
                        f"전체 평균 {overall:.1%}; 높은 관절: " + ", ".join(f"{j} {v:.0%}" for j, v in top.items())
                        + " (state 1=inferred 해석은 [E])"))

    # B5 Orientation 구조적 0값
    def _b5() -> Fig:
        jz = jo.sort_values(["group", "subject_id", "exercise"]).copy()
        jz["sess"] = jz["subject_id"] + " Ex" + jz["exercise"].astype(str)
        sess_order = list(dict.fromkeys(jz["sess"]))
        m = jz.pivot(index="joint", columns="sess", values="zero_frame_ratio").reindex(index=list(io.JOINTS),
                                                                                      columns=sess_order)
        fig, ax = plt.subplots(figsize=(15, 6.5))
        im = ax.imshow(m.values.astype(float), cmap=SEQ, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(io.JOINTS)),
                      [f"{j} †" if j in io.NO_ORIENTATION_JOINTS else j for j in io.JOINTS], fontsize=8)
        ax.set_xticks(range(len(sess_order)), sess_order, rotation=90, fontsize=5.5)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("4값 모두 0인 frame 비율", color=INK2)
        ax.set_xlabel("† = 원저자 Script 주석 'no orientation' 관절", color=INK2)
        always = sorted(jz.groupby("joint")["zero_frame_ratio"].min().loc[lambda s: s == 1].index,
                        key=io.JOINTS.index)
        match = set(always) == set(io.NO_ORIENTATION_JOINTS)
        return (Fig(_save(fig, out, "B5_orientation_zero_joints",
                              f"B5. Orientation 구조적 0값 — 관절 × Session (n={len(sess_order)})"), "B",
                        "Orientation 구조적 0값", "Orientation의 구조적 0값은 어떤 관절에 존재하는가?",
                        f"모든 Session·frame에서 0인 관절: {', '.join(always)} → Script 주석과 "
                        + ("일치" if match else "불일치")))

    # B6 파일 누락 Group × Exercise
    def _b6() -> Fig:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
        for ax, (c, lab) in zip(axes, [("has_position", "Position 없음"), ("has_timestamp", "Timestamp 없음"),
                                       ("has_orientation", "Orientation 없음")]):
            miss = es.assign(m=~es[c].astype(bool))
            k = miss.pivot_table(index="group", columns="exercise", values="m", aggfunc="sum", observed=False).reindex(GROUP_ORDER)
            n = miss.pivot_table(index="group", columns="exercise", values="m", aggfunc="size", observed=False).reindex(GROUP_ORDER)
            txt = np.array([[f"{_n(a)}/{_n(b)}" for a, b in zip(r1, r2)] for r1, r2 in zip(k.values, n.values)])
            _annot_heatmap(ax, (k / n).values.astype(float), txt, GROUP_ORDER, EX_LABELS, cmap=SEQ_WARN, vmax=1)
            ax.set_title(lab, fontsize=10)
        miss_rate = (~es["has_position"].astype(bool)).groupby(es["group"], observed=False).mean().reindex(GROUP_ORDER)
        return (Fig(_save(fig, out, "B6_missing_files_group_exercise", "B6. 파일 누락 — Group × Exercise (칸: 누락/격자)"), "B",
                        "파일 누락 분포", "파일 누락은 어느 Group/Exercise에 집중되는가?",
                        "Position 누락률: " + ", ".join(f"{g} {v:.0%}" for g, v in miss_rate.items())))

    for tag, fn, dep in (("B1", _b1, ts), ("B2", _b2, ts), ("B3", _b3, ts), ("B4", _b4, js), ("B5", _b5, jo), ("B6", _b6, None)):
        if dep is not None and dep.empty:
            figs.append(_placeholder(out, tag, "신호 파일 없음 (읽을 수 있는 파일 0개)"))
        else:
            figs.append(fn())
    return figs


# ------------------------------------------------------- C. clinical scores


def fig_scores(res, out) -> list[Fig]:
    es = res["es"]
    sc = es[es["C1_has_clinical_file"].astype(bool)]
    figs = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, s in zip(axes, labels.SCORES):
        lo, hi = labels.RANGES[s]
        _box_strip(ax, sc, "group", s, GROUP_ORDER, colors=GROUP_COLORS)
        ax.set_ylim(lo - 1, hi + 2)
        ax.set_title(f"{s} (범위 {lo}~{hi}, Session 단위)", fontsize=10)
        ax.set_ylabel(s)
    med = sc.groupby("group", observed=False)["TS"].median().reindex(GROUP_ORDER)
    figs.append(Fig(_save(fig, out, "C1_scores_by_group", "C1. Group별 TS / PO / CF 분포"), "C",
                    "Group별 점수 분포", "TS/PO/CF의 Group별 분포는? (같은 Subject의 5 Session이 함께 그려짐)",
                    "TS 중앙값: " + ", ".join(f"{g} {v:.1f}" for g, v in med.items())))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, s in zip(axes, labels.SCORES):
        lo, hi = labels.RANGES[s]
        _box_strip(ax, sc, "exercise", s, list(labels.EXERCISES), color_by="group", colors=GROUP_COLORS)
        ax.set_xticklabels([f"Ex{e}\nn={_n(sc.loc[sc['exercise'] == e, s].notna().sum())}" for e in labels.EXERCISES])
        ax.set_ylim(lo - 1, hi + 2)
        ax.set_title(s, fontsize=10)
    _group_legend(axes[2], loc="lower left")
    figs.append(Fig(_save(fig, out, "C2_scores_by_exercise", "C2. Exercise별 TS / PO / CF 분포 (점 색 = Group)"), "C",
                    "Exercise별 점수 분포", "Ex1~Ex5별 점수 분포는?"))

    fig, axes = plt.subplots(1, 3, figsize=(17, 3.9), sharey=True)
    for ax, s in zip(axes, labels.SCORES):
        lo, hi = labels.RANGES[s]
        med = sc.pivot_table(index="group", columns="exercise", values=s, aggfunc="median", observed=False).reindex(GROUP_ORDER)
        mean = sc.pivot_table(index="group", columns="exercise", values=s, aggfunc="mean", observed=False).reindex(GROUP_ORDER)
        n = sc.pivot_table(index="group", columns="exercise", values=s, aggfunc="count", observed=False).reindex(GROUP_ORDER)
        txt = np.array([[f"중앙 {a:.1f}\n평균 {b:.1f}\nn={_n(c)}" if c == c and c else "n=0"
                         for a, b, c in zip(r1, r2, r3)] for r1, r2, r3 in zip(med.values, mean.values, n.values)])
        _annot_heatmap(ax, med.values.astype(float), txt, GROUP_ORDER, EX_LABELS, vmin=lo, vmax=hi, fig=fig,
                       cbar_label=f"{s} 중앙값")
        ax.set_title(s, fontsize=10)
    figs.append(Fig(_save(fig, out, "C3_scores_group_exercise_heatmap", "C3. Group × Exercise 점수 (색: 중앙값, 칸: 중앙·평균·n)"),
                    "C", "Group × Exercise 점수", "Group × Exercise 조합별 대표 점수와 표본 수는?"))
    return figs


# ------------------------------------------------------------ D. ceiling


def fig_ceiling(res, out) -> list[Fig]:
    es, t = res["es"], res["tables"]
    ts = es[es["TS"].notna()]
    figs = []
    bins = np.arange(0, 51, 2.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].hist(ts["TS"], bins=bins, color="#2a78d6", edgecolor="white")
    axes[0].axvline(CEILING_TS, color=ALERT, ls="--", lw=1.2)
    k = int((ts["TS"] >= CEILING_TS).sum())
    axes[0].set_title(f"전체 Session n={len(ts)}, TS ≥ {CEILING_TS:g}: {k} ({k / len(ts):.0%})", fontsize=10)
    axes[0].set_xlabel("TS")
    axes[0].set_ylabel("Session 수")
    for c in COHORT_ORDER:
        v = ts.loc[ts["cohort"] == c, "TS"]
        axes[1].hist(v, bins=bins, histtype="step", lw=2, color=COHORT_COLORS[c], label=f"{c} (n={len(v)})",
                     density=True)
    axes[1].axvline(CEILING_TS, color=ALERT, ls="--", lw=1.2)
    axes[1].set_xlabel("TS")
    axes[1].set_ylabel("밀도 (군 내 비율)")
    axes[1].legend(loc="upper left")
    axes[1].set_title("건강군 vs 환자군", fontsize=10)
    figs.append(Fig(_save(fig, out, "D1_ts_histogram", "D1. TS 분포와 천장 구간 (TS ≥ 45)"), "D",
                    "TS histogram", "TS 분포는 어떤 모양이고, 상단(≥45)에 얼마나 몰려 있는가?",
                    f"전체 {k}/{len(ts)} Session이 TS ≥ {CEILING_TS:g}"))

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharex=True, sharey=True)
    for ax, g in zip(axes, GROUP_ORDER):
        v = ts.loc[ts["group"] == g, "TS"]
        ax.hist(v, bins=bins, color=GROUP_COLORS[g], edgecolor="white")
        ax.axvline(CEILING_TS, color=ALERT, ls="--", lw=1)
        kk = int((v >= CEILING_TS).sum())
        ax.set_title(f"{g}\nn={len(v)}, ≥45: {kk / len(v):.0%}" if len(v) else f"{g}\nn=0", fontsize=10)
        ax.set_xlabel("TS")
    axes[0].set_ylabel("Session 수")
    figs.append(Fig(_save(fig, out, "D2_ts_histogram_by_group", "D2. Group별 TS histogram"), "D",
                    "Group별 TS histogram", "Group마다 TS 분포 모양과 천장 구간 비중은 어떻게 다른가?"))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    _box_strip(ax, ts, "group", "TS", GROUP_ORDER, colors=GROUP_COLORS, ref_lines=((CEILING_TS, "TS = 45"),))
    ax.set_ylim(-1, 52)
    ax.set_ylabel("TS")
    figs.append(Fig(_save(fig, out, "D3_ts_box_by_group", "D3. Group별 TS boxplot (점선: TS = 45)"), "D",
                    "Group별 TS boxplot", "Group별 TS 중앙값·분산과 천장 근접 정도는?"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    cg = t["ceiling_by_group"].set_index("group").reindex(GROUP_ORDER)
    x = np.arange(len(cg))
    bars = axes[0].bar(x, cg["ratio"], color=[GROUP_COLORS[g] for g in GROUP_ORDER], width=0.6)
    axes[0].errorbar(x, cg["ratio"], yerr=[cg["ratio"] - cg["ci_low"], cg["ci_high"] - cg["ratio"]],
                     fmt="none", ecolor=INK2, capsize=4, lw=1)
    for b, (_, r_) in zip(bars, cg.iterrows()):
        if not r_["n"] > 0:  # 해당 Group 에 TS 값이 없음
            axes[0].text(b.get_x() + b.get_width() / 2, 0.02, "n=0", ha="center", va="bottom", fontsize=8)
            continue
        axes[0].text(b.get_x() + b.get_width() / 2, r_["ci_high"] + 0.02, f"{r_['ratio']:.0%}\n{_n(r_['k_ts_ge_45'])}/{_n(r_['n'])}",
                     ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, GROUP_ORDER)
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0].set_ylabel("TS ≥ 45 Session 비율")
    axes[0].set_title("Group별 (막대 = Wilson 95% CI, Session 단위·Subject 내 상관 미반영)", fontsize=9)
    axes[0].grid(axis="x", visible=False)
    ge = t["ceiling_by_group_exercise"]
    r_ = ge.pivot(index="group", columns="exercise", values="ratio").reindex(GROUP_ORDER)
    kk = ge.pivot(index="group", columns="exercise", values="k_ts_ge_45").reindex(GROUP_ORDER)
    nn = ge.pivot(index="group", columns="exercise", values="n").reindex(GROUP_ORDER)
    txt = np.array([[f"{a:.0%}\n{_n(b)}/{_n(c)}" if c == c else "n=0" for a, b, c in zip(r1, r2, r3)]
                    for r1, r2, r3 in zip(r_.values, kk.values, nn.values)])
    _annot_heatmap(axes[1], r_.values.astype(float), txt, GROUP_ORDER, EX_LABELS, cmap=SEQ_WARN, vmax=1,
                   fig=fig, cbar_label="TS ≥ 45 비율")
    axes[1].set_title("Group × Exercise — 천장효과가 심한 운동", fontsize=10)
    cc = t["ceiling_by_cohort"].set_index("cohort")
    ce = t["ceiling_by_cohort_exercise"]
    h = ce[ce["cohort"] == COHORT_ORDER[0]].sort_values("ratio", ascending=False)
    finding = (f"건강군 {cc.loc[COHORT_ORDER[0], 'ratio']:.0%} "
               f"({_n(cc.loc[COHORT_ORDER[0], 'k_ts_ge_45'])}/{_n(cc.loc[COHORT_ORDER[0], 'n'])}) · "
               f"환자군 {cc.loc[COHORT_ORDER[1], 'ratio']:.0%} "
               f"({_n(cc.loc[COHORT_ORDER[1], 'k_ts_ge_45'])}/{_n(cc.loc[COHORT_ORDER[1], 'n'])}); "
               "건강군 운동별: " + ", ".join(f"Ex{_n(e)} {v:.0%}" for e, v in zip(h["exercise"], h["ratio"])))
    figs.append(Fig(_save(fig, out, "D4_ts_ge45_ratio", "D4. TS ≥ 45 비율 — Group별, Group × Exercise"), "D",
                    "TS ≥ 45 비율", "건강군에서 TS ≥ 45가 얼마나 되는가? 천장효과가 어느 Exercise에서 심한가?", finding))
    return figs


# ------------------------------------------------------------- E. label QC


def fig_label_qc(res, out) -> list[Fig]:
    es, t = res["es"], res["tables"]
    figs = []

    lm = t["label_missing"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    for ax, s in zip(axes[:3], labels.SCORES):
        d = lm[lm["target"] == s]
        k = d.pivot(index="group", columns="exercise", values="n_missing_in_c1").reindex(GROUP_ORDER)
        n = d.pivot(index="group", columns="exercise", values="n_c1").reindex(GROUP_ORDER)
        txt = np.array([[f"{_n(a)}/{_n(b)}" for a, b in zip(r1, r2)] for r1, r2 in zip(k.values, n.values)])
        _annot_heatmap(ax, (k / n.replace(0, np.nan)).values.astype(float), txt, GROUP_ORDER, EX_LABELS,
                       cmap=SEQ_WARN, vmax=0.5)
        ax.set_title(f"{s} 값 결측 (C1 대상자 중)", fontsize=10)
    nofile = lm[(lm["target"] == "TS") & (lm["exercise"] == 1)].set_index("group").reindex(GROUP_ORDER)
    bars = axes[3].barh(GROUP_ORDER, nofile["n_no_clinical_file"], color=[GROUP_COLORS[g] for g in GROUP_ORDER])
    for b, (_, r_) in zip(bars, nofile.iterrows()):
        axes[3].text(b.get_width(), b.get_y() + b.get_height() / 2,
                     f" {_n(r_['n_no_clinical_file'])}/{_n(r_['n_subjects'])}", va="center", fontsize=8)
    axes[3].invert_yaxis()
    axes[3].set_title("Clinical 파일 자체가 없는 Subject", fontsize=10)
    axes[3].grid(axis="y", visible=False)
    miss = es[es["C1_has_clinical_file"].astype(bool) & es[list(labels.SCORES)].isna().any(axis=1)]
    where = miss.groupby("subject_id")["exercise"].apply(lambda v: ",".join(f"Ex{e}" for e in v))
    figs.append(Fig(_save(fig, out, "E1_label_missing_heatmap", "E1. Label 결측 — Group × Exercise × Target"), "E",
                    "Label 결측", "결측은 어디에 있는가? (파일 없음 vs 파일 내 값 없음 구분)",
                    "파일 내 결측: " + ("; ".join(f"{s} {v}" for s, v in where.items()) or "없음")))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    n_out_total = 0
    for ax, s in zip(axes, labels.SCORES):
        lo, hi = labels.RANGES[s]
        d = es[es[s].notna()]
        oor = d[f"{s}_out_of_range"].astype(bool)
        n_out_total += int(oor.sum())
        ax.axhspan(lo, hi, color="#eaf2fc", zorder=0)
        ax.scatter(_jitter(int((~oor).sum()), 7, 0.3), d.loc[~oor, s], s=12, color="#2a78d6", alpha=0.6,
                   edgecolors="white", linewidths=0.4)
        if oor.any():
            ax.scatter(_jitter(int(oor.sum()), 8, 0.3), d.loc[oor, s], s=40, color=ALERT, marker="X", label="범위 이탈")
            ax.legend()
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        ax.set_ylim(min(lo, d[s].min()) - 2, max(hi, d[s].max()) + 2)
        ax.set_title(f"{s}: 허용 {lo}~{hi} · 이탈 {_n(oor.sum())}건 (n={len(d)})", fontsize=10)
        ax.axhline(lo, color=INK2, lw=0.8)
        ax.axhline(hi, color=INK2, lw=0.8)
    figs.append(Fig(_save(fig, out, "E2_label_range", "E2. Label 범위 검사 (파란 띠 = 허용 범위, X = 이탈)"), "E",
                    "범위 이탈", "TS 0~50, PO 0~15, CF 0~35 범위를 벗어난 값이 있는가?",
                    f"범위 이탈 {n_out_total}건"))

    d = es[es["delta"].notna()]
    nz = d[d["delta_nonzero"].astype(bool)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    # 부동소수 오차(1e-12 수준)가 0 근처 막대를 둘로 쪼개지 않도록 0을 중심으로 칸을 나눈다.
    step = 0.05
    lim = max(1.0, float(d["delta"].abs().max()) * 1.2)
    edges = np.arange(-lim - step / 2, lim + step, step)
    dv = d["delta"].where(d["delta"].abs() > labels.SUM_TOL, 0.0)
    axes[0].hist(dv, bins=edges, color="#2a78d6", edgecolor="none")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.5, len(d) * 3)
    axes[0].set_xlabel(f"delta = TS − (PO + CF)   (|delta| ≤ {labels.SUM_TOL} 은 0으로 표시)")
    axes[0].set_ylabel("Session 수 (log)")
    axes[0].set_title(f"n={len(d)}, |delta| > {labels.SUM_TOL}: {len(nz)}건", fontsize=10)
    for off, (dval, g_) in enumerate(nz.groupby(nz["delta"].round(6))):
        lab = ", ".join(f"{r_['subject_id']} Ex{_n(r_['exercise'])}" for _, r_ in g_.iterrows())
        axes[0].annotate(lab, (dval, len(g_)), xytext=(-20, 40 + 14 * off), textcoords="offset points",
                         ha="right", fontsize=8, color=ALERT, arrowprops={"arrowstyle": "-", "color": ALERT})
    axes[1].axis("off")
    if len(nz):
        cell = [[r_["subject_id"], r_["group"], f"Ex{_n(r_['exercise'])}", f"{r_['TS']:.3f}", f"{r_['PO']:.3f}",
                 f"{r_['CF']:.3f}", f"{r_['delta']:+.3f}"] for _, r_ in nz.iterrows()]
        tb = axes[1].table(cellText=cell, colLabels=["Subject", "Group", "Ex", "TS", "PO", "CF", "delta"],
                           loc="center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.5)
        tb.scale(1, 1.4)
        axes[1].set_title("delta ≠ 0 인 Session", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "delta ≠ 0 인 Session 없음", ha="center")
    figs.append(Fig(_save(fig, out, "E3_ts_minus_po_cf", "E3. TS − (PO + CF) 차이"), "E",
                    "TS와 PO+CF 차이", "TS = PO + CF 가 성립하지 않는 Session은?",
                    "; ".join(f"{r_['subject_id']} Ex{_n(r_['exercise'])} ({r_['delta']:+.2f})" for _, r_ in nz.iterrows())
                    or "없음"))
    return figs


# ------------------------------------------------------- F. value format


def fig_format(res, out) -> list[Fig]:
    t, vals = res["tables"], res["format_values"]
    figs = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.3]})

    pv = t["format_per_value"].groupby(["group", "format"], observed=True)["n"].sum().unstack(fill_value=0)
    pv = pv.reindex(index=GROUP_ORDER, columns=FORMAT_ORDER, fill_value=0)
    ps = t["format_per_subject"].pivot(index="group", columns="value_format", values="n")
    ps = ps.reindex(index=GROUP_ORDER, columns=FORMAT_ORDER).fillna(0)
    for ax, tab, lab in ((axes[0], pv, "값 단위 (TS·PO·CF 전체)"), (axes[1], ps, "Subject 단위 (C2, 우선순위: 불규칙>1/3>정수)")):
        left = np.zeros(len(tab))
        for f in FORMAT_ORDER:
            v = tab[f].to_numpy(dtype=float)
            ax.barh(GROUP_ORDER, v, left=left, color=FORMAT_COLORS[f], label=FORMAT_KO[f], edgecolor="white", lw=1.5)
            for i, (l_, w_) in enumerate(zip(left, v)):
                if w_ > 0:
                    ax.text(l_ + w_ / 2, i, f"{_n(w_)}", ha="center", va="center", fontsize=8, color="white")
            left += v
        ax.invert_yaxis()
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=8)

    ax = axes[2]
    for i, g in enumerate(GROUP_ORDER):
        d = vals[vals["group"] == g]
        ax.scatter(d["frac_part"], i + _jitter(len(d), i, 0.25), s=10, alpha=0.6,
                   c=[FORMAT_COLORS[f] for f in d["format"]], edgecolors="none")
    for x in (0, 1 / 3, 2 / 3):
        ax.axvline(x, color=INK2, ls=":", lw=0.8)
    ax.set_xticks([0, 1 / 3, 2 / 3, 1], ["0", "1/3", "2/3", "1"])
    ax.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("소수부 (값 − floor(값))")
    ax.set_title("값의 소수부 분포 — 1/3 격자 밖이면 불규칙", fontsize=10)
    irr = t["format_irregular_subjects"]
    tot = ps.sum()
    figs.append(Fig(_save(fig, out, "F1_label_value_format", "F1. Clinical score 값 형식 (정수 / 1/3 단위 / 불규칙 소수)"), "F",
                    "값 형식", "점수가 정수·1/3 단위·불규칙 소수 중 무엇인가? 불규칙 값은 어떤 대상자인가?",
                    f"C2 Subject: 정수 {_n(tot['integer'])} · 1/3 {_n(tot['third'])} · 불규칙 {_n(tot['irregular'])} "
                    f"({', '.join(irr['subject_id'])})"))
    return figs


# --------------------------------------------------------------- G. age


def fig_age(res, out) -> list[Fig]:
    subjects, es, t = res["interim"]["subjects"], res["es"], res["tables"]
    s = subjects.copy()
    s["cohort"] = np.where(s["group_top"] == "CG", COHORT_ORDER[0], COHORT_ORDER[1])
    a = s[s["has_age"].astype(bool)]
    ag = t["age_by_group"].set_index("group")
    figs = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    _box_strip(axes[0], a, "group", "age", GROUP_ORDER, colors=GROUP_COLORS)
    axes[0].set_xticklabels([f"{g}\nAge n={_n(ag.loc[g, 'n_age'])} / 결측 {_n(ag.loc[g, 'n_age_missing'])}"
                             for g in GROUP_ORDER])
    axes[0].set_ylabel("Age")
    axes[0].set_title("Group별 Age (SuppInfo에 Age가 있는 대상자만)", fontsize=10)
    x = np.arange(len(GROUP_ORDER))
    axes[1].bar(x, ag["n_age"], color=[GROUP_COLORS[g] for g in GROUP_ORDER], label="Age 있음", edgecolor="white")
    axes[1].bar(x, ag["n_age_missing"], bottom=ag["n_age"], color=ABSENT, label="Age 결측 (SuppInfo 없음)",
                edgecolor=INK2, hatch="///", lw=0.5)
    for i, g in enumerate(GROUP_ORDER):
        axes[1].text(i, ag.loc[g, "n_subjects"], f"{_n(ag.loc[g, 'n_age_missing'])} 결측", ha="center",
                     va="bottom", fontsize=8)
    axes[1].set_xticks(x, GROUP_ORDER)
    axes[1].set_ylabel("Subject 수")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", visible=False)
    figs.append(Fig(_save(fig, out, "G1_age_by_group", "G1. Group별 Age 분포와 Age 결측"), "G",
                    "Group별 Age", "Group별 Age 분포는? Age 결측은 몇 명인가?",
                    f"Age 있음 {_n(ag['n_age'].sum())}/{_n(ag['n_subjects'].sum())}명, 결측 {_n(ag['n_age_missing'].sum())}명 "
                    "(결측 값은 보정하지 않음)"))

    co = t["age_cohort"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.arange(10, 95, 5)
    for c in COHORT_ORDER:
        v = a.loc[a["cohort"] == c, "age"]
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.35, color=COHORT_COLORS[c], edgecolor=COHORT_COLORS[c],
                lw=2, label=f"{c} n={len(v)}, 중앙 {v.median():.0f}세")
    ax.set_xlabel("Age")
    ax.set_ylabel("Subject 수")
    ax.legend()
    ax.set_title(f"Cliff's δ (환자군 vs 건강군) = {co['cliffs_delta_patient_vs_healthy']:+.2f}  "
                 "(|δ|≥0.474 = 큰 효과)", fontsize=10)
    figs.append(Fig(_save(fig, out, "G2_age_healthy_vs_patient", "G2. 건강군 vs 환자군 Age 분포"), "G",
                    "건강군 vs 환자군 Age", "건강군과 환자군의 Age 분포가 얼마나 다른가?",
                    f"중앙값 건강군 {co['median_healthy']:.0f}세 vs 환자군 {co['median_patient']:.0f}세, "
                    f"Cliff's δ = {co['cliffs_delta_patient_vs_healthy']:+.2f}"))

    corr = t["age_score_spearman"]
    fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True)
    for i, sc_ in enumerate(labels.SCORES):
        lo, hi = labels.RANGES[sc_]
        for j, ex in enumerate(labels.EXERCISES):
            ax = axes[i, j]
            d = es[(es["exercise"] == ex) & es["age"].notna() & es[sc_].notna()]
            for c in COHORT_ORDER:
                dc = d[d["cohort"] == c]
                ax.scatter(dc["age"], dc[sc_], s=16, color=COHORT_COLORS[c], alpha=0.75, edgecolors="white",
                           linewidths=0.5, label=c)
            cc = corr[(corr["exercise"] == ex) & (corr["target"] == sc_)].set_index("scope")
            txt = "\n".join(f"{k}: ρ={cc.loc[k, 'spearman_rho']:+.2f} (n={_n(cc.loc[k, 'n'])})"
                            for k in ("전체", *COHORT_ORDER))
            ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=7, va="bottom", color=INK,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
            ax.set_ylim(lo - 1, hi + 1)
            if i == 0:
                ax.set_title(f"Ex{ex}", fontsize=10)
            if j == 0:
                ax.set_ylabel(sc_)
            if i == 2:
                ax.set_xlabel("Age")
    axes[0, 4].legend(loc="upper right", fontsize=8)
    ts_c = corr[(corr["target"] == "TS")].pivot(index="exercise", columns="scope", values="spearman_rho")
    figs.append(Fig(_save(fig, out, "G3_age_vs_scores", "G3. Age와 TS / PO / CF (Spearman ρ: 전체 vs 군 내부)"), "G",
                    "Age와 점수의 관계", "Age와 TS/PO/CF는 어떤 관계인가? 군 내부에서도 유지되는가? (Session 단위 탐색, 다중비교 보정 없음. Subject 단위 ρ·p값과 검정은 docs/label_eda_report.md §4.3·§6)",
                    "TS ρ 전체: " + ", ".join(f"Ex{e} {v:+.2f}" for e, v in ts_c["전체"].items())
                    + " / 건강군 내: " + ", ".join(f"Ex{e} {v:+.2f}" for e, v in ts_c[COHORT_ORDER[0]].items())
                    + " / 환자군 내: " + ", ".join(f"Ex{e} {v:+.2f}" for e, v in ts_c[COHORT_ORDER[1]].items())))
    return figs


# ------------------------------------------------------------------- all


def make_all(res, out: str | Path) -> list[Fig]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figs = []
    for fn in (fig_overview, fig_quality, fig_scores, fig_ceiling, fig_label_qc, fig_format, fig_age):
        figs += fn(res, out)
    return figs


SECTION_TITLES = {
    "A": "A. Dataset Overview (데이터 구조)",
    "B": "B. 데이터 품질 (Raw 신호 QC)",
    "C": "C. Clinical Score 분포",
    "D": "D. 천장효과",
    "E": "E. Label QC",
    "F": "F. Label 값 형식",
    "G": "G. Age / Confounding",
}


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "- 없음"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) and not float(v).is_integer() else
                 (str(int(v)) if isinstance(v, float) and v == v else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(res, figs: list[Fig], path: str | Path) -> None:
    path = Path(path)
    t = res["tables"]
    sig = res["signal"]
    parts = [
        "# KIMORE EDA · 데이터 품질 보고서 (자동 생성)",
        "",
        f"> 생성: `python -m kimore.eda` · {date.today().isoformat()}",
        f"> **{PRELIM}** — 공식 원본 재확보 후 같은 코드로 다시 생성한다. 이 파일은 코드가 덮어쓴다.",
        "> 입력: `data/interim/` (`python -m kimore.qc` 산출물) + `data/raw/` 읽기 전용.",
        "> 범위: EDA·데이터 품질 확인만. ML·CV·feature 확정 없음. 통계는 기술통계·Spearman·효과크기(탐색적).",
        "> Session 단위 그래프에서 같은 Subject의 여러 Session은 독립 표본이 아니다.",
        "> **역할 분담**: 이 보고서는 데이터 구조·신호 품질·Label QC 시각화와 점수·천장·Age 의 Session 단위 탐색 그래프를 맡는다. "
        "Subject 단위 검정(Kruskal-Wallis·Mann-Whitney), 불규칙 소수 라벨 민감도, 연령 overlap·매칭, 주 분석 후보, Gate 2 판정은 "
        "[docs/label_eda_report.md](../../docs/label_eda_report.md) (`python -m kimore.label_eda`) 에 있다. "
        "천장 기준·비율, TS−(PO+CF), 상관은 두 보고서가 같은 함수(kimore.label_eda)로 계산한다.",
        "",
    ]
    for sec in SECTION_TITLES:
        parts += [f"## {SECTION_TITLES[sec]}", ""]
        for f in [f for f in figs if f.section == sec]:
            parts += [f"### {f.title}", f"- **질문**: {f.question}"]
            if f.finding:
                parts.append(f"- **계산 결과**: {f.finding}")
            parts += ["", f"![{f.title}](figures/{f.file})", ""]
        if sec == "B":
            sk = sig["skipped"]
            parts += ["### 신호 파일을 읽지 않은 Session (자동 선택하지 않음 / 파싱 실패)", _md_table(sk), ""]
        if sec == "D":
            parts += ["### TS ≥ 45 표 (Group)",
                      _md_table(t["ceiling_by_group"][["group", "n", "k_ts_ge_45", "ratio", "ci_low", "ci_high",
                                                       "n_max_TS", "n_max_PO", "n_max_CF"]]), ""]
        if sec == "E":
            parts += ["### delta ≠ 0 표", _md_table(t["delta_nonzero"]), "",
                      "### 범위 이탈 표", _md_table(t["out_of_range"]), ""]
        if sec == "F":
            parts += ["### 불규칙 소수 대상자", _md_table(t["format_irregular_subjects"]), ""]
        if sec == "G":
            parts += ["### Group별 Age", _md_table(t["age_by_group"], "{:.1f}"), ""]
    parts += [
        "## 산출물",
        "- 그래프: `reports/eda/figures/*.png`",
        "- 통계표: `reports/eda/tables/*.csv`",
        "- EDA 정리 데이터: `data/interim/eda/eda_sessions.csv` (Subject × Exercise 격자 + 점수 + Age + 파생 QC 열), "
        "`signal_*.csv` (Timestamp 간격, 관절별 tracking state, 관절별 Orientation 0값)",
        "",
        "## 해석상 주의",
        "- tracking state 1 = inferred, Timestamp 단위, 1/3 단위의 의미는 모두 미확인 [E] 이다.",
        "- Wilson CI 와 Spearman ρ 는 Session 을 독립으로 가정한 탐색적 수치이며, 다중비교 보정을 하지 않았다.",
        "- Group 차이에는 Age 차이가 섞여 있다(G 섹션). Group 차이를 곧바로 운동 수행 능력 차이로 해석하지 않는다.",
        "- 위 이상값(delta ≠ 0, 불규칙 소수 등)은 오류로 단정하지 않는다. 원본 재확보 후 다시 확인한다.",
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")
