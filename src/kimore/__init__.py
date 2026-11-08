"""KIMORE 재활운동 데이터 처리 패키지.

모듈
- io: 원본 CSV 파싱과 Kinect v2 25-joint schema
- labels: ClinicalAssessment / SuppInfo 로딩과 Label QC
- qc: 원본 폴더 인벤토리와 manifest 생성 (`python -m kimore.qc`)
- cv: Subject 단위 분할 규칙 검증
"""
