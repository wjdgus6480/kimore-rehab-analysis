"""인벤토리·manifest 구조 테스트 (REQUIREMENTS §4.3, §14).

- 합성 원본 트리로 탐지 규칙을 검증한다.
- 실제 data/raw 가 있으면 수치가 아닌 불변 조건만 검사한다 (없으면 skip).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pytest

from kimore import io, labels, qc

ROOT = Path(__file__).resolve().parents[1]
REAL_RAW = ROOT / "data" / "raw"


# ------------------------------------------------------------------ fixtures


def _write_signal(path: Path, n_frames: int, n_cols: int, fill=0.5, blank=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = ",".join([str(fill)] * n_cols) + ","
    path.write_text("\n" * blank + "\n".join([row] * n_frames) + "\n", encoding="ascii")


def _write_position(path: Path, n_frames: int, state: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    joint = f"0.1,0.2,2.5,{state}"
    row = ",".join([joint] * io.N_JOINTS) + ","
    path.write_text("\n\n" + "\n".join([row] * n_frames) + "\n", encoding="ascii")


def _write_timestamp(path: Path, n_frames: int, step: int = 330000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{1000 + i * step}," for i in range(n_frames)) + "\n", encoding="ascii")


def _write_clinical(path: Path, sid: str, values: dict[str, float | None]):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    cols = labels.score_columns()
    ws.append(["Subject ID"] + [f"clinical {c.split('_')[0]} Ex#{c[-1]}" for c in cols])
    ws.append([sid] + [values.get(c) for c in cols])
    wb.save(path)


def _write_supp(path: Path, sid: str, group: str, age):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Subject ID", "Group", "Age", "Gender"])
    ws.append([sid, group, age, "F"])
    wb.save(path)


def _full_scores(ts=45.0, po=13.0, cf=32.0) -> dict:
    out = {}
    for e in labels.EXERCISES:
        out[f"TS_Ex{e}"], out[f"PO_Ex{e}"], out[f"CF_Ex{e}"] = ts, po, cf
    return out


@pytest.fixture()
def raw(tmp_path: Path) -> Path:
    r = tmp_path / "raw"
    e1 = r / "CG" / "Expert" / "E_ID1"
    e2 = r / "CG" / "Expert" / "E_ID2"
    s1 = r / "GPP" / "Stroke" / "S_ID1"

    # E_ID1: Es1 정상 녹화, Es2 Timestamp 없음, Es3 프레임 수 불일치, Es4 폴더 없음
    _write_position(e1 / "Es1" / "Raw" / "JointPosition_a.csv", 10)
    _write_timestamp(e1 / "Es1" / "Raw" / "TimeStamp_a.csv", 10)
    _write_position(e1 / "Es2" / "Raw" / "JointPosition_b.csv", 10)
    _write_position(e1 / "Es3" / "Raw" / "JointPosition_c.csv", 10)
    _write_timestamp(e1 / "Es3" / "Raw" / "TimeStamp_c.csv", 9)
    (e1 / "Es5").mkdir(parents=True)
    _write_clinical(e1 / "Es1" / "Label" / "ClinicalAssessment_E_ID1.xlsx", "E_ID1", _full_scores())
    _write_clinical(e1 / "Es2" / "Label" / "ClinicalAssessment_E_ID1(1).xlsx", "E_ID1", _full_scores())
    _write_supp(e1 / "Es1" / "Label" / "SuppInfo_E_ID1.xlsx", "E_ID1", "E", 30)

    # E_ID2: 라벨 없음, 대신 E_ID1 의 라벨 파일이 잘못 놓여 있음, 녹화 2개
    _write_position(e2 / "Es1" / "Raw" / "JointPosition_x.csv", 5)
    _write_timestamp(e2 / "Es1" / "Raw" / "TimeStamp_x.csv", 5)
    _write_position(e2 / "Es1" / "Raw" / "JointPosition_y.csv", 6)
    _write_timestamp(e2 / "Es1" / "Raw" / "TimeStamp_y.csv", 6)
    _write_supp(e2 / "Es2" / "Label" / "SuppInfo_E_ID1.xlsx", "E_ID1", "E", 30)

    # S_ID1: 점수 결측·범위 이탈·합 불일치, Age 결측, tracking state 이상값
    scores = _full_scores(30.0, 10.0, 20.0)
    scores["TS_Ex2"] = None
    scores["PO_Ex3"] = 16.0
    scores["TS_Ex4"] = 31.0
    _write_clinical(s1 / "Es1" / "Label" / "ClinicalAssessment_S_ID1.xlsx", "S_ID1", scores)
    _write_supp(s1 / "Es1" / "Label" / "SuppInfo_S_ID1.xlsx", "S_ID1", "S", None)
    _write_position(s1 / "Es1" / "Raw" / "JointPosition_s.csv", 4, state=7)
    _write_timestamp(s1 / "Es1" / "Raw" / "TimeStamp_s.csv", 4)
    return r


@pytest.fixture()
def result(raw: Path, tmp_path: Path):
    return qc.run(raw, tmp_path / "out", tmp_path / "report.md")


def _session(sessions, sid, ex):
    hit = sessions[(sessions.subject_id == sid) & (sessions.exercise == ex)]
    assert len(hit) == 1
    return hit.iloc[0]


# ------------------------------------------------------------- synthetic tests


def test_outputs_written(result, tmp_path):
    out = tmp_path / "out"
    for name in ("manifest_files.csv", "manifest_sessions.csv", "subjects.csv",
                 "labels_clinical_long.csv", "label_qc_issues.csv"):
        assert (out / name).is_file()
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# KIMORE 인벤토리")


def test_session_grid_is_subject_x_exercise(result):
    s, subj = result["sessions"], result["subjects"]
    assert len(s) == len(subj) * len(labels.EXERCISES)
    assert not s.duplicated(["subject_id", "exercise"]).any()


def test_session_status_rules(result):
    s = result["sessions"]
    assert _session(s, "E_ID1", 1)["qc_status"] == "ok"
    assert _session(s, "E_ID1", 2)["qc_status"] == "excluded"          # Timestamp 없음
    assert "Timestamp 없음" in _session(s, "E_ID1", 2)["qc_reasons"]
    assert _session(s, "E_ID1", 3)["qc_status"] == "warning"           # 프레임 수 불일치
    assert "프레임 수 불일치" in _session(s, "E_ID1", 3)["qc_reasons"]
    assert not _session(s, "E_ID1", 4)["session_folder_exists"]
    assert _session(s, "E_ID1", 4)["qc_status"] == "excluded"
    assert _session(s, "E_ID2", 1)["n_recordings"] == 2                # 자동 선택 안 함
    assert _session(s, "E_ID2", 1)["qc_status"] == "warning"
    assert "0/1/2 외" in _session(s, "S_ID1", 1)["qc_reasons"]
    # Es3 Position 은 합성 데이터라 Es1 과 내용이 같다 → 숨기지 않고 경고
    assert "내용이 같음" in _session(s, "E_ID1", 3)["qc_reasons"]
    assert _session(s, "E_ID1", 3)["has_position"]


def test_every_non_ok_session_has_reason(result):
    s = result["sessions"]
    assert (s.loc[s.qc_status != "ok", "qc_reasons"].str.len() > 0).all()


def test_copy_and_misplaced_detection(result):
    f = result["files"].set_index("name")
    assert f.loc["ClinicalAssessment_E_ID1(1).xlsx", "is_copy_name"]
    assert f.loc["ClinicalAssessment_E_ID1(1).xlsx", "duplicate_of"] is not None
    misplaced = result["files"][result["files"].misplaced]
    assert list(misplaced["path"]) == ["CG/Expert/E_ID2/Es2/Label/SuppInfo_E_ID1.xlsx"]


def test_subject_counts_use_normalized_ids(result):
    subj = result["subjects"].set_index("subject_id")
    # 복사본이 있어도 E_ID1 은 한 번만 센다
    assert subj.index.is_unique
    assert subj.loc["E_ID1", "C1_has_clinical_file"]
    assert subj.loc["E_ID1", "clin_n_files"] == 2
    assert not subj.loc["E_ID1", "clin_conflict"]
    # 잘못 놓인 파일은 폴더 주인(E_ID2)이 아니라 파일명 주인(E_ID1)에 속한다
    assert not subj.loc["E_ID2", "has_suppinfo"]
    # Age 결측은 보정하지 않는다
    assert subj.loc["S_ID1", "has_suppinfo"] and not subj.loc["S_ID1", "has_age"]
    assert np.isnan(subj.loc["S_ID1", "age"])


def test_criteria_are_nested(result):
    subj = result["subjects"]
    assert (subj.C3_has_all_scores <= subj.C2_has_any_score).all()
    assert (subj.C2_has_any_score <= subj.C1_has_clinical_file).all()


def test_label_issues_detected(result):
    got = {(r.subject_id, r.exercise, r.check) for r in result["issues"].itertuples()}
    assert ("S_ID1", 2, "missing") in got
    assert ("S_ID1", 3, "out_of_range") in got
    assert ("S_ID1", 4, "sum_mismatch") in got
    assert not any(sid == "E_ID1" for sid, _, _ in got)


def test_raw_is_not_modified(raw: Path, tmp_path: Path):
    before = {p: p.stat().st_mtime_ns for p in raw.rglob("*") if p.is_file()}
    qc.run(raw, tmp_path / "out2")
    after = {p: p.stat().st_mtime_ns for p in raw.rglob("*") if p.is_file()}
    assert before == after


# ------------------------------------------------------- real data invariants


@pytest.mark.skipif(not REAL_RAW.is_dir(), reason="data/raw 없음")
def test_real_data_invariants(tmp_path: Path):
    res = qc.run(REAL_RAW, tmp_path / "out", compute_hash=False)
    s, subj, f = res["sessions"], res["subjects"], res["files"]

    assert subj.subject_id.is_unique
    assert set(subj.group) <= set(qc.GROUPS)
    assert subj.id_prefix_ok.all()
    assert len(s) == len(subj) * len(labels.EXERCISES)
    assert not s.duplicated(["subject_id", "exercise"]).any()
    assert set(s.qc_status) <= {"ok", "warning", "excluded"}
    assert (~s.has_position <= (s.qc_status == "excluded")).all()
    assert (~s.has_timestamp <= (s.qc_status == "excluded")).all()
    assert (s.loc[s.qc_status != "ok", "qc_reasons"].str.len() > 0).all()
    assert (subj.C3_has_all_scores <= subj.C2_has_any_score).all()
    assert (subj.C2_has_any_score <= subj.C1_has_clinical_file).all()
    assert (subj.has_age <= subj.has_suppinfo).all()
    ok = s[s.qc_status == "ok"]
    assert (ok.n_frames > 0).all()
    assert f.loc[f.kind.isin(["clinical", "suppinfo"]), "label_subject_id"].notna().all()
