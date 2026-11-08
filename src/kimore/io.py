"""KIMORE 원본 CSV 파싱과 Kinect v2 25-joint schema.

근거 (REQUIREMENTS §3.2, §3.3, §7.1)
- 관절 순서와 이름: 원저자 preproc_Ex*.m 의 index_<Joint> 정의 [S]
- 'no orientation' 관절: 같은 Script의 주석 [S]
- 파일 형식(헤더 없음, 앞쪽 빈 행, 행 끝 쉼표): 데이터에서 확인 [D]
- Orientation 4값의 성분 순서는 미확인이므로 q0~q3으로만 이름 붙인다.
- Timestamp 단위는 미확인이므로 원시값 그대로 둔다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS: tuple[str, ...] = (
    "Spine_Base",
    "Spine_Mid",
    "Neck",
    "Head",
    "Shoulder_Left",
    "Elbow_Left",
    "Wrist_Left",
    "Hand_Left",
    "Shoulder_Right",
    "Elbow_Right",
    "Wrist_Right",
    "Hand_Right",
    "Hip_Left",
    "Knee_Left",
    "Ankle_Left",
    "Foot_Left",
    "Hip_Right",
    "Knee_Right",
    "Ankle_Right",
    "Foot_Right",
    "Spine_Shoulder",
    "Tip_Left",
    "Thumb_Left",
    "Tip_Right",
    "Thumb_Right",
)
N_JOINTS = len(JOINTS)

NO_ORIENTATION_JOINTS: tuple[str, ...] = (
    "Head",
    "Foot_Left",
    "Foot_Right",
    "Tip_Left",
    "Thumb_Left",
    "Tip_Right",
    "Thumb_Right",
)

POSITION_FIELDS: tuple[str, ...] = ("x", "y", "z", "state")
ORIENTATION_FIELDS: tuple[str, ...] = ("q0", "q1", "q2", "q3")

SUBJECT_ID_RE = re.compile(r"^(?:E|NE|B|P|S)_ID\d+$")
COPY_SUFFIX_RE = re.compile(r"\(\d+\)(?=\.[^.]+$)")
LABEL_FILE_RE = re.compile(
    r"^(?P<kind>ClinicalAssessment|SuppInfo)_(?P<sid>(?:E|NE|B|P|S)_ID\d+)(?P<copy>\(\d+\))?\.xlsx$"
)
RECORDING_KEY_RE = re.compile(r"^(?:JointPosition|JointOrientation|TimeStamp)(?P<key>.*?)(?:\(\d+\))?\.csv$")


class ParseError(ValueError):
    """원본 파일이 기대한 형식이 아닐 때 발생한다."""


@dataclass(frozen=True)
class ParseInfo:
    n_lines: int
    n_blank: int
    n_frames: int
    trailing_comma: str  # "all" | "none" | "mixed"


def position_columns() -> list[str]:
    return [f"{j}_{f}" for j in JOINTS for f in POSITION_FIELDS]


def orientation_columns() -> list[str]:
    return [f"{j}_{f}" for j in JOINTS for f in ORIENTATION_FIELDS]


def _read_table(path: str | Path, n_cols: int) -> tuple[np.ndarray, ParseInfo]:
    path = Path(path)
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise ParseError(f"{path.name}: ASCII가 아닌 문자가 있습니다") from exc

    lines = text.splitlines()
    data_lines = [ln.strip() for ln in lines if ln.strip().strip(",") != ""]
    if not data_lines:
        raise ParseError(f"{path.name}: 데이터 행이 없습니다")

    ends = [ln.endswith(",") for ln in data_lines]
    trailing = "all" if all(ends) else "none" if not any(ends) else "mixed"

    rows = []
    for i, ln in enumerate(data_lines, start=1):
        fields = ln.split(",")
        if fields[-1] == "":
            fields = fields[:-1]
        if len(fields) != n_cols:
            raise ParseError(f"{path.name}: {i}번째 데이터 행의 열 수 {len(fields)} (기대 {n_cols})")
        rows.append(fields)

    try:
        arr = np.array(rows, dtype=float)
    except ValueError as exc:
        raise ParseError(f"{path.name}: 숫자가 아닌 값이 있습니다") from exc
    if not np.isfinite(arr).all():
        raise ParseError(f"{path.name}: NaN 또는 무한대 값이 있습니다")

    info = ParseInfo(
        n_lines=len(lines),
        n_blank=len(lines) - len(data_lines),
        n_frames=len(data_lines),
        trailing_comma=trailing,
    )
    return arr, info


def read_position(path: str | Path) -> tuple[pd.DataFrame, ParseInfo]:
    arr, info = _read_table(path, N_JOINTS * len(POSITION_FIELDS))
    return pd.DataFrame(arr, columns=position_columns()), info


def read_orientation(path: str | Path) -> tuple[pd.DataFrame, ParseInfo]:
    arr, info = _read_table(path, N_JOINTS * len(ORIENTATION_FIELDS))
    return pd.DataFrame(arr, columns=orientation_columns()), info


def read_timestamp(path: str | Path) -> tuple[pd.Series, ParseInfo]:
    arr, info = _read_table(path, 1)
    values = arr[:, 0]
    if np.all(values == np.round(values)):
        return pd.Series(values.astype(np.int64), name="timestamp_raw"), info
    return pd.Series(values, name="timestamp_raw"), info


def frames_aligned(*n_frames: int) -> bool:
    """여러 파일의 프레임 수가 모두 같은지."""
    return len(set(n_frames)) <= 1


def is_copy_name(filename: str) -> bool:
    """'name(1).ext' 처럼 복사본 접미사가 붙은 파일명인지."""
    return COPY_SUFFIX_RE.search(filename) is not None


def parse_label_filename(filename: str) -> tuple[str, str, bool] | None:
    """라벨 파일명에서 (종류, 정규화된 Subject ID, 복사본 여부)를 꺼낸다."""
    m = LABEL_FILE_RE.match(filename)
    if not m:
        return None
    return m.group("kind"), m.group("sid"), m.group("copy") is not None


def recording_key(filename: str) -> str | None:
    """JointPosition011214_103748.csv → '011214_103748'. 같은 녹화의 파일끼리 묶는 데 쓴다."""
    m = RECORDING_KEY_RE.match(filename)
    return m.group("key") if m else None
