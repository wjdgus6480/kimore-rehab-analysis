"""원본 폴더 인벤토리와 data manifest 생성 (REQUIREMENTS §4.3~§4.6, Gate 1).

사용:
    python -m kimore.qc --raw data/raw --out data/interim --report docs/inventory_report.md

산출물 (out 폴더)
- manifest_files.csv     파일 단위: 종류, 위치, 크기, SHA-256, 복사본·중복·잘못 놓인 파일
- manifest_sessions.csv  Session(Subject × Exercise) 단위: 파일 존재, 프레임 수, 신호 QC, qc_status
- subjects.csv           Subject 단위: SuppInfo·Age, 집계 기준 C1~C3, 라벨 값 형식, 라벨 충돌
- labels_clinical_long.csv  Subject × Exercise 의 TS/PO/CF
- label_qc_issues.csv    결측·범위·TS≠PO+CF

원칙
- data/raw 는 읽기만 한다.
- 값을 보정하거나 추정하지 않는다. 판단이 필요한 것은 qc_status 사유로만 남긴다.
- Timestamp 단위는 미확인이므로 원시값 간격만 기록한다.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from kimore import io, labels

GROUPS: dict[str, tuple[str, str]] = {
    # group: (상위 폴더, Subject ID 접두사)
    "Expert": ("CG", "E"),
    "NotExpert": ("CG", "NE"),
    "BackPain": ("GPP", "B"),
    "Parkinson": ("GPP", "P"),
    "Stroke": ("GPP", "S"),
}
EXERCISE_DIRS = tuple(f"Es{i}" for i in labels.EXERCISES)
RAW_KINDS = ("position", "orientation", "timestamp")

_KIND_RULES = (
    ("position", re.compile(r"^JointPosition.*\.csv$")),
    ("orientation", re.compile(r"^JointOrientation.*\.csv$")),
    ("timestamp", re.compile(r"^TimeStamp.*\.csv$")),
    ("clinical", re.compile(r"^ClinicalAssessment_.*\.xlsx$")),
    ("suppinfo", re.compile(r"^SuppInfo_.*\.xlsx$")),
    ("script", re.compile(r"\.m$")),
    ("video", re.compile(r"\.(avi|mp4)$", re.I)),
    ("shortcut", re.compile(r"\.lnk$", re.I)),
)


def file_kind(name: str) -> str:
    for kind, rule in _KIND_RULES:
        if rule.search(name):
            return kind
    return "other"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- files


def scan_files(raw_root: str | Path, compute_hash: bool = True) -> pd.DataFrame:
    """raw_root 아래 모든 파일을 한 행씩 기록한다."""
    raw_root = Path(raw_root)
    rows = []
    for path in sorted(p for p in raw_root.rglob("*") if p.is_file()):
        rel = path.relative_to(raw_root)
        parts = rel.parts
        name = path.name
        label = io.parse_label_filename(name)
        rows.append(
            {
                "path": rel.as_posix(),
                "group_top": parts[0] if len(parts) > 1 else None,
                "group": parts[1] if len(parts) > 2 else None,
                "subject_folder": parts[2] if len(parts) > 3 else None,
                "exercise_folder": parts[3] if len(parts) > 4 and parts[3] in EXERCISE_DIRS else None,
                "subfolder": parts[4] if len(parts) > 5 else None,
                "name": name,
                "kind": file_kind(name),
                "is_copy_name": io.is_copy_name(name),
                "label_subject_id": label[1] if label else None,
                "recording_key": io.recording_key(name),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path) if compute_hash else None,
            }
        )
    files = pd.DataFrame(rows)
    if files.empty:
        return files

    files["misplaced"] = files["label_subject_id"].notna() & (
        files["label_subject_id"] != files["subject_folder"]
    )

    # 내용이 같은 파일: 이름에 복사본 접미사가 없는 것을 원본으로 삼는다.
    files["duplicate_of"] = None
    if compute_hash:
        for _, grp in files.groupby(["kind", "sha256"]):
            if len(grp) < 2:
                continue
            ordered = grp.sort_values(["is_copy_name", "path"])
            canonical = ordered.iloc[0]["path"]
            files.loc[ordered.index[1:], "duplicate_of"] = canonical
    return files


# ------------------------------------------------------------------------ subjects


def expected_subjects(raw_root: str | Path) -> pd.DataFrame:
    """폴더 구조에서 Subject 목록을 만든다 (Group 은 폴더 기준)."""
    raw_root = Path(raw_root)
    rows = []
    for group, (top, prefix) in GROUPS.items():
        gdir = raw_root / top / group
        if not gdir.is_dir():
            continue
        for sdir in sorted(d for d in gdir.iterdir() if d.is_dir()):
            rows.append(
                {
                    "subject_id": sdir.name,
                    "group": group,
                    "group_top": top,
                    "id_prefix_ok": sdir.name.startswith(prefix + "_ID") and bool(io.SUBJECT_ID_RE.match(sdir.name)),
                }
            )
    return pd.DataFrame(rows)


def _read_label_files(files: pd.DataFrame, raw_root: Path, kind: str, reader) -> pd.DataFrame:
    """라벨 파일을 모두 읽어 (subject_id, path, 내용...) 행으로 만든다. 읽기 실패는 error 열에 남긴다."""
    rows = []
    for rec in files[files["kind"] == kind].itertuples():
        row = {"subject_id": rec.label_subject_id, "path": rec.path, "error": None}
        try:
            row.update(reader(raw_root / rec.path))
        except Exception as exc:  # noqa: BLE001 - 어떤 실패든 manifest 에 기록
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def _content_signature(row: pd.Series, cols: list[str]) -> tuple:
    out = []
    for c in cols:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        elif isinstance(v, float):
            out.append(round(v, 9))
        else:
            out.append(str(v))
    return tuple(out)


def build_subjects(raw_root: Path, files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Subject 표, 대표 Clinical wide 표, 라벨 파일별 읽기 결과를 만든다."""
    subjects = expected_subjects(raw_root).set_index("subject_id")
    score_cols = labels.score_columns()

    clin_files = _read_label_files(files, raw_root, "clinical", labels.read_clinical)
    supp_files = _read_label_files(files, raw_root, "suppinfo", labels.read_suppinfo)

    def summarise(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
        recs = {}
        if df.empty:
            return pd.DataFrame()
        for sid, grp in df.groupby("subject_id"):
            ok = grp[grp["error"].isna()]
            sigs = {_content_signature(r, cols) for _, r in ok.iterrows()}
            first = ok.iloc[0] if len(ok) else None
            recs[sid] = {
                f"{prefix}_n_files": len(grp),
                f"{prefix}_n_read_errors": int(grp["error"].notna().sum()),
                f"{prefix}_conflict": len(sigs) > 1,
                **({c: first[c] for c in cols} if first is not None else {}),
            }
        return pd.DataFrame.from_dict(recs, orient="index")

    supp_cols = ["subject_in_file", "group", "age", "gender"]
    supp = summarise(supp_files, supp_cols, "supp").rename(
        columns={"subject_in_file": "supp_subject_in_file", "group": "supp_group", "age": "age", "gender": "gender"}
    )
    clin = summarise(clin_files, ["subject_in_file", *score_cols], "clin").rename(
        columns={"subject_in_file": "clin_subject_in_file"}
    )

    subjects = subjects.join(supp, how="left").join(clin, how="left")
    subjects["has_suppinfo"] = subjects.get("supp_n_files", pd.Series(dtype=float)).fillna(0) > 0
    subjects["has_age"] = subjects["has_suppinfo"] & subjects.get("age", pd.Series(dtype=float)).notna()

    present = subjects.reindex(columns=score_cols).notna().sum(axis=1)
    subjects["n_scores_present"] = present
    subjects["C1_has_clinical_file"] = subjects.get("clin_n_files", pd.Series(dtype=float)).fillna(0) > 0
    subjects["C2_has_any_score"] = present >= 1
    subjects["C3_has_all_scores"] = present == len(score_cols)
    subjects["value_format"] = [
        labels.subject_value_format(list(r)) for r in subjects.reindex(columns=score_cols).itertuples(index=False)
    ]

    # 라벨 파일에 있는데 폴더에는 없는 Subject 는 따로 표시한다.
    orphan_ids = set(clin_files.get("subject_id", [])) | set(supp_files.get("subject_id", []))
    orphan_ids -= set(subjects.index)
    if orphan_ids:
        print(f"[warn] 폴더에 없는 Subject ID 의 라벨 파일: {sorted(orphan_ids)}")

    wide = subjects.loc[subjects["C1_has_clinical_file"], score_cols]
    label_files = pd.concat(
        [clin_files.assign(kind="clinical"), supp_files.assign(kind="suppinfo")], ignore_index=True
    )
    return subjects.reset_index().rename(columns={"index": "subject_id"}), wide, label_files


# ------------------------------------------------------------------------ sessions


def _timestamp_stats(ts: pd.Series) -> dict:
    d = np.diff(ts.to_numpy(dtype=float))
    if len(d) == 0:
        return {"ts_median_interval_raw": np.nan, "ts_n_duplicate": 0, "ts_n_backward": 0, "ts_n_gap": 0}
    med = float(np.median(d))
    return {
        "ts_median_interval_raw": med,
        "ts_n_duplicate": int((d == 0).sum()),
        "ts_n_backward": int((d < 0).sum()),
        # 단위 미확인: 중앙값의 1.5배를 넘는 간격을 '프레임 누락 의심'으로만 센다.
        "ts_n_gap": int((d > 1.5 * med).sum()) if med > 0 else 0,
    }


def _position_stats(pos: pd.DataFrame) -> dict:
    state = pos[[f"{j}_state" for j in io.JOINTS]].to_numpy()
    values = set(np.unique(state).tolist())
    return {
        "state_values": ",".join(str(int(v)) for v in sorted(values)),
        "state_unexpected": bool(values - {0, 1, 2}),
        "state1_ratio": float((state == 1).mean()),
        "state0_ratio": float((state == 0).mean()),
    }


def _orientation_stats(ori: pd.DataFrame) -> dict:
    zero = [
        j
        for j in io.JOINTS
        if (ori[[f"{j}_{f}" for f in io.ORIENTATION_FIELDS]].to_numpy() == 0).all()
    ]
    return {
        "ori_all_zero_joints": ",".join(zero),
        "ori_zero_matches_script": set(zero) == set(io.NO_ORIENTATION_JOINTS),
    }


def build_sessions(raw_root: Path, files: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    """Subject × Exercise 격자 전체를 만들고 각 Session 의 원본 파일을 점검한다."""
    raw_root = Path(raw_root)
    score_cols = labels.score_columns()
    subj = subjects.set_index("subject_id")
    # '(n)' 이름의 복사본만 제외한다. 다른 세션과 내용이 같은 원본은 숨기지 않고 경고로 남긴다.
    is_copy = files["is_copy_name"] & files["duplicate_of"].notna()
    raw = files[files["kind"].isin(RAW_KINDS) & ~is_copy]

    rows = []
    for sid, srow in subj.iterrows():
        for ex_i, ex_dir in zip(labels.EXERCISES, EXERCISE_DIRS):
            folder = raw_root / srow["group_top"] / srow["group"] / sid / ex_dir
            sel = raw[(raw["subject_folder"] == sid) & (raw["exercise_folder"] == ex_dir)]
            row = {
                "subject_id": sid,
                "group": srow["group"],
                "exercise": ex_i,
                "session_folder_exists": folder.is_dir(),
                "has_position": (sel["kind"] == "position").any(),
                "has_orientation": (sel["kind"] == "orientation").any(),
                "has_timestamp": (sel["kind"] == "timestamp").any(),
                "has_clinical": bool(srow.get("C1_has_clinical_file", False)),
                "has_suppinfo": bool(srow.get("has_suppinfo", False)),
                "has_scores": all(pd.notna(srow.get(f"{s}_Ex{ex_i}")) for s in labels.SCORES),
                "n_recordings": sel["recording_key"].nunique(),
            }
            reasons: list[str] = []
            status = "ok"

            if not row["session_folder_exists"]:
                status, reasons = "excluded", ["운동 폴더 없음"]
            else:
                for kind in RAW_KINDS:
                    if (sel["kind"] == kind).sum() > 1:
                        reasons.append(f"{kind} 파일 {int((sel['kind'] == kind).sum())}개")
                if not row["has_position"]:
                    status = "excluded"
                    reasons.append("Position 없음")
                if not row["has_timestamp"]:
                    status = "excluded"
                    reasons.append("Timestamp 없음")
                if row["n_recordings"] > 1:
                    status = "excluded" if status == "excluded" else "warning"
                    reasons.append("녹화 여러 개 (자동 선택하지 않음)")

            if sel["duplicate_of"].notna().any():
                if status == "ok":
                    status = "warning"
                reasons.append("다른 위치의 원본 파일과 내용이 같음")

            if status != "excluded" and row["n_recordings"] == 1:
                status, reasons = _check_recording(raw_root, sel, row, status, reasons)
            elif row["has_position"] and row["n_recordings"] == 1:
                # Timestamp 가 없어도 Position 자체의 형식·품질은 기록한다.
                status, reasons = _check_recording(raw_root, sel, row, status, reasons)

            if not row["has_orientation"] and row["session_folder_exists"]:
                reasons.append("Orientation 없음(필수 아님)")
            if not row["has_scores"]:
                reasons.append("해당 운동 점수 없음")

            row["qc_status"] = status
            row["qc_reasons"] = "; ".join(reasons)
            rows.append(row)

    sessions = pd.DataFrame(rows)
    sessions["analyzable_signal"] = sessions["qc_status"] != "excluded"
    sessions["analyzable_with_scores"] = sessions["analyzable_signal"] & sessions["has_scores"]
    return sessions


def _check_recording(raw_root: Path, sel: pd.DataFrame, row: dict, status: str, reasons: list[str]):
    frames = {}
    for kind, reader, stats in (
        ("position", io.read_position, _position_stats),
        ("orientation", io.read_orientation, _orientation_stats),
        ("timestamp", io.read_timestamp, _timestamp_stats),
    ):
        hit = sel[sel["kind"] == kind]
        if hit.empty:
            continue
        path = raw_root / hit.iloc[0]["path"]
        try:
            data, info = reader(path)
        except io.ParseError as exc:
            if kind in ("position", "timestamp"):
                status = "excluded"
            elif status == "ok":
                status = "warning"
            reasons.append(f"{kind} 파싱 실패: {exc}")
            continue
        frames[kind] = info.n_frames
        row[f"{kind}_n_rows_raw"] = info.n_lines
        row[f"{kind}_n_blank"] = info.n_blank
        row[f"{kind}_n_frames"] = info.n_frames
        row.update(stats(data))

    if frames and not io.frames_aligned(*frames.values()):
        if status == "ok":
            status = "warning"
        reasons.append("프레임 수 불일치 " + str(frames))
    row["n_frames"] = frames.get("position", np.nan)
    if row.get("state_unexpected"):
        status = "warning" if status == "ok" else status
        reasons.append("tracking state 에 0/1/2 외 값")
    if row.get("ts_n_backward", 0) or row.get("ts_n_duplicate", 0):
        status = "warning" if status == "ok" else status
        reasons.append("Timestamp 역행·중복")
    if "ori_zero_matches_script" in row and not row["ori_zero_matches_script"]:
        status = "warning" if status == "ok" else status
        reasons.append("Orientation 0값 관절이 Script 주석과 다름")
    return status, reasons


# -------------------------------------------------------------------------- report


def _count_table(df: pd.DataFrame, cols: list[str], by: str = "group") -> pd.DataFrame:
    order = [g for g in GROUPS if g in set(df[by])]
    table = df.groupby(by)[cols].sum().reindex(order).astype(int)
    table.loc["합계"] = table.sum()
    return table


def _md(table: pd.DataFrame) -> str:
    cols = [table.index.name or ""] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    def fmt(v) -> str:
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    for idx, *values in table.itertuples(name=None):
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in values]) + " |")
    return "\n".join(lines)


def write_report(path: Path, files, subjects, sessions, issues, label_files) -> None:
    s = subjects.assign(n_subjects=1)
    subj_tab = _count_table(
        s, ["n_subjects", "has_suppinfo", "has_age", "C1_has_clinical_file", "C2_has_any_score", "C3_has_all_scores"]
    )
    subj_tab.index.name = "Group"
    sess_tab = _count_table(
        sessions.assign(n_sessions=1),
        ["n_sessions", "session_folder_exists", "has_position", "has_timestamp", "has_orientation",
         "analyzable_signal", "analyzable_with_scores"],
    )
    sess_tab.index.name = "Group"
    status = sessions.groupby(["group", "qc_status"]).size().unstack(fill_value=0)
    status = status.reindex([g for g in GROUPS if g in status.index])
    status.index.name = "Group"
    fmt = subjects[subjects["C2_has_any_score"]].groupby(["group", "value_format"]).size().unstack(fill_value=0)
    fmt.index.name = "Group"
    ana = sessions[sessions["analyzable_with_scores"]].groupby(["group", "exercise"]).size().unstack(fill_value=0)
    ana.index.name = "Group"

    copies = files[files["is_copy_name"]]
    misplaced = files[files["misplaced"]]
    raw_dup = files[files["kind"].isin(RAW_KINDS) & files["duplicate_of"].notna()]
    conflicts = subjects[subjects.get("clin_conflict", False).fillna(False).astype(bool)
                         | subjects.get("supp_conflict", False).fillna(False).astype(bool)]
    read_errors = label_files[label_files["error"].notna()]
    age = subjects[subjects["has_age"]].groupby("group")["age"].agg(["count", "mean", "min", "max"])
    age = age.reindex([g for g in GROUPS if g in age.index]).round(1)
    age.index.name = "Group"

    kinds = files.groupby("kind").agg(n=("path", "size"), MB=("size_bytes", lambda x: round(x.sum() / 1e6, 1)))
    kinds.index.name = "kind"

    issue_tab = issues.groupby("check").size().to_frame("n") if len(issues) else pd.DataFrame({"n": []})
    issue_tab.index.name = "check"

    def listing(df, cols):
        return "\n".join(f"- `{' | '.join(str(r[c]) for c in cols)}`" for _, r in df.iterrows()) or "- 없음"

    text = f"""# KIMORE 인벤토리 보고서 (자동 생성)

> 생성: `python -m kimore.qc` · {date.today().isoformat()}
> **개발용·예비 결과** — 현재 로컬 사본 기준이며 공식 원본 재확보 후 다시 생성한다.
> 이 파일은 코드가 덮어쓴다. 직접 수정하지 말 것.

## 1. 파일 종류
{_md(kinds)}

## 2. Subject (집계 기준은 REQUIREMENTS §4.6)
- C1 = ClinicalAssessment 파일 있음, C2 = 점수 1개 이상, C3 = 15개 점수 모두. 복사본 제외, Subject ID 기준.

{_md(subj_tab)}

### Age (Age 가 있는 대상자만)
{_md(age)}

### 라벨 값 형식 (C2 대상자)
{_md(fmt)}

## 3. Session (Subject × Exercise 격자)
{_md(sess_tab)}

### qc_status
{_md(status)}

### 신호 + 점수가 모두 있는 Session (운동별)
{_md(ana) if len(ana) else "- 없음"}

## 4. Label QC
{_md(issue_tab) if len(issue_tab) else "- 문제 없음"}

{listing(issues, ["subject_id", "exercise", "check", "detail"])}

## 5. 파일 이상
### 복사본 접미사 `(n)` 파일 ({len(copies)})
{listing(copies, ["path", "duplicate_of"])}

### 다른 Subject 폴더에 놓인 라벨 파일 ({len(misplaced)})
{listing(misplaced, ["path", "label_subject_id"])}

### 내용이 같은 원본 신호 파일 ({len(raw_dup)})
{listing(raw_dup, ["path", "duplicate_of"])}

### 같은 Subject 의 라벨 파일끼리 내용 불일치 ({len(conflicts)})
{listing(conflicts, ["subject_id"])}

### 라벨 파일 읽기 실패 ({len(read_errors)})
{listing(read_errors, ["path", "error"])}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------- main


def run(raw_root: str | Path, out_dir: str | Path, report: str | Path | None = None, compute_hash: bool = True):
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = scan_files(raw_root, compute_hash=compute_hash)
    subjects, wide, label_files = build_subjects(raw_root, files)
    sessions = build_sessions(raw_root, files, subjects)
    issues = labels.label_qc(wide)
    long = labels.clinical_long(wide)

    files.to_csv(out_dir / "manifest_files.csv", index=False, encoding="utf-8-sig")
    sessions.to_csv(out_dir / "manifest_sessions.csv", index=False, encoding="utf-8-sig")
    subjects.to_csv(out_dir / "subjects.csv", index=False, encoding="utf-8-sig")
    long.to_csv(out_dir / "labels_clinical_long.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(out_dir / "label_qc_issues.csv", index=False, encoding="utf-8-sig")
    if report:
        write_report(Path(report), files, subjects, sessions, issues, label_files)
    return {"files": files, "subjects": subjects, "sessions": sessions, "issues": issues, "label_files": label_files}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="KIMORE 원본 인벤토리·manifest 생성 (원본은 읽기만 함)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--report", default="docs/inventory_report.md")
    ap.add_argument("--no-hash", action="store_true", help="SHA-256 계산 생략 (중복 탐지도 생략됨)")
    args = ap.parse_args(argv)
    res = run(args.raw, args.out, args.report, compute_hash=not args.no_hash)
    s = res["sessions"]
    print(
        f"files={len(res['files'])} subjects={len(res['subjects'])} sessions={len(s)} "
        f"analyzable_signal={int(s['analyzable_signal'].sum())} "
        f"analyzable_with_scores={int(s['analyzable_with_scores'].sum())} label_issues={len(res['issues'])}"
    )


if __name__ == "__main__":
    main()
