"""
config.py — 결과 테이블 채우기 설정
모든 경로/매핑/선택 옵션을 이 파일에서만 수정하면 됩니다.
"""
from pathlib import Path

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────

# total (master) 파일 경로
MASTER_PATH = Path(
    "/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
    "/reports/yolo_conf_0.01_iou_0.5/master.xlsx"
)

# format 템플릿 파일 경로 (행/열 구조가 정의된 빈 테이블)
FORMAT_PATH = Path(
    "/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
    "/src/utils/table/detection_table_format.xlsx"
)

# 출력 파일 경로 (master.xlsx와 같은 폴더)
OUTPUT_PATH = MASTER_PATH.parent / "detection_results.xlsx"


# ──────────────────────────────────────────────
# test dataset 선택 (3_5, 4 시트용)
# ──────────────────────────────────────────────
# 선택지: "FGART" | "DDR" | "E-optha" | "IDRiD" | "DIARETDB1" | "ALL"
# "ALL" 선택 시 → 시트명이 "{base_sheet_name}_{TEST}" 형태로 5개 생성

TESTSET_3_5 = "FGART"
TESTSET_4   = "FGART"


# ──────────────────────────────────────────────
# test dataset 레이블 → master 파일 블록 헤더 키워드
# (master 파일 내 블록 감지에 사용)
# ──────────────────────────────────────────────
TEST_BLOCK_KEYWORDS = {
    "FGART":     "FGART",
    "DDR":       "DDR",
    "E-optha":   "eophtha",
    "IDRiD":     "IDRiD",
    "DIARETDB1": "DIARETDB1",
}

ALL_TESTSETS = list(TEST_BLOCK_KEYWORDS.keys())


# ──────────────────────────────────────────────
# 모델 매핑: 시트별로 (format Train축, Method축) → master 시트명
#
# format 파일의 시트마다 Train·Method 레이블이 다르게 표기됨:
#   2_main_acc/det : Train="Merged",  Method="YOLOv12"
#   3_5_class_acc_det: Train="Merge", Method="YOLOv12"
#   4_FPPI_FROC    : Train="Merge",  Method="YOLO12"
#
# 나중에 모델 추가 시 각 MAP에 한 줄씩 추가.
# "4cls": 4개 클래스 시트명 (MA/HE/EX/SE + overall)
# "1cls": 1개 클래스 시트명 (lesion)
# None  : 해당 조합 데이터 없음 → 공란 유지
# ──────────────────────────────────────────────

# 2_main_acc / 2_main_det 용
MODEL_SHEET_MAP_2MAIN = {
    ("FGART",  "YOLOv12"):  {"4cls": "fgart_yolo12",      "1cls": "fgart_yolo12_1cls"},
    ("DDR",    "YOLOv12"):  {"4cls": "ddr_yolo12",        "1cls": "ddr_yolo12_1cls"},
    ("Merged", "YOLOv12"):  {"4cls": "merge_crop_yolo12", "1cls": "merge_crop_yolo12_1cls"},
    ("E-optha","YOLOv12"):  {"4cls": None,                "1cls": "eophtha_yolo12_1cls"},
    # 추가 예시:
    # ("FGART",  "RT-DETR"): {"4cls": "fgart_rtdetr", "1cls": "fgart_rtdetr_1cls"},
}

# 3_5_class_acc_det 용 (Train: "Merge" / "E ophtha")
MODEL_SHEET_MAP_35 = {
    ("FGART",    "YOLOv12"): {"4cls": "fgart_yolo12",      "1cls": "fgart_yolo12_1cls"},
    ("DDR",      "YOLOv12"): {"4cls": "ddr_yolo12",        "1cls": "ddr_yolo12_1cls"},
    ("Merge",    "YOLOv12"): {"4cls": "merge_crop_yolo12", "1cls": "merge_crop_yolo12_1cls"},
    ("E ophtha", "YOLOv12"): {"4cls": None,                "1cls": "eophtha_yolo12_1cls"},
    # 추가 예시:
    # ("FGART", "RT-DETR"): {"4cls": "fgart_rtdetr", "1cls": "fgart_rtdetr_1cls"},
}

# 4_FPPI_FROC 용 (Train: "Merge" / "E ophtha", Method: "YOLO12")
MODEL_SHEET_MAP_4 = {
    ("FGART",    "YOLO12"): {"4cls": "fgart_yolo12",      "1cls": "fgart_yolo12_1cls"},
    ("DDR",      "YOLO12"): {"4cls": "ddr_yolo12",        "1cls": "ddr_yolo12_1cls"},
    ("Merge",    "YOLO12"): {"4cls": "merge_crop_yolo12", "1cls": "merge_crop_yolo12_1cls"},
    ("E ophtha", "YOLO12"): {"4cls": None,                "1cls": "eophtha_yolo12_1cls"},
    # 추가 예시:
    # ("FGART", "RT-DETR"): {"4cls": "fgart_rtdetr", "1cls": "fgart_rtdetr_1cls"},
}


# ──────────────────────────────────────────────
# format 파일 셀 위치 매핑 (수정 불필요)
# ──────────────────────────────────────────────

# 2_main_acc / 2_main_det: test dataset → col 시작 위치
TEST_COL_START = {
    "FGART":     3,   # col C
    "DDR":       6,   # col F
    "E-optha":   9,   # col I
    "IDRiD":    12,   # col L
    "DIARETDB1":15,   # col O
}

# 2_main_acc metrics: col offset (test_col_start 기준)
ACC_COL_OFFSET = {
    "detection_acc":     0,  # +0 = Loc. Acc.
    "classification_acc":1,  # +1 = Cls. Acc.
    "overall_acc":       2,  # +2 = Overall Acc.
}

# 2_main_det metrics: col offset
DET_COL_OFFSET = {
    "AP@0.5":   0,  # +0 = mAP
    "precision":1,  # +1 = P.
    "recall":   2,  # +2 = R.
}

# 3_5_class_acc_det: 고정 col 위치
COL_35 = {
    "AP_MA":       3,
    "AP_HE":       4,
    "AP_EX":       5,
    "AP_SE":       6,
    "mAP05":       7,
    "mAP095":      8,
    "cls_acc_MA":  9,
    "cls_acc_HE": 10,
    "cls_acc_EX": 11,
    "cls_acc_SE": 12,
    "cls_acc_mean":13,
    "1cls_mAP05": 14,
    "1cls_mAP095":15,
}

# 4_FPPI_FROC: 고정 col 위치
COL_4 = {
    "FPPI=0.125": 3,
    "FPPI=0.25":  4,
    "FPPI=0.5":   5,
    "FPPI=1.0":   6,
    "FPPI=2.0":   7,
    "FPPI=4.0":   8,
    "FPPI=8.0":   9,
    "avg_FROC":  10,
}
