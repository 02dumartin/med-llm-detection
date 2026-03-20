#!/usr/bin/env python3
"""
IDRiD 원본 데이터 → COCO JSON 변환 스크립트.

- 분할 없음: 전체 81장을 test로만 출력
- 4cls: MA=0, HE=1, EX=2, SE=3 (0-based, Optic Disc 제외)
- IDRiD 마스크는 0/76 값 사용 → binary = (mask > 0)

입력: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD (또는 --idrid-root)
  - A. Segmentation/1. Original Images/{a. Training Set, b. Testing Set}
  - A. Segmentation/2. All Segmentation Groundtruths/{a. Training Set, b. Testing Set}/{1..4 lesion folders}

출력:
  - 4cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_4cls/test/test.json + images/
  - 1cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_1cls/test/test.json + images/
"""

from pathlib import Path
import json
import argparse

import cv2
import numpy as np
from PIL import Image

# 경로
DEFAULT_IDRID_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD")
OUT_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_4cls")
OUT_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_1cls")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MIN_AREA = 25
MIN_BOX_SIZE = 64  # FGART와 동일: 너무 작은 bbox는 중심 고정 후 최소 크기로 확장

# 4cls: MA=0, HE=1, EX=2, SE=3 (DDR/FGART 통일)
CATEGORIES_4CLS = [
    {"id": 0, "name": "MA"},
    {"id": 1, "name": "HE"},
    {"id": 2, "name": "EX"},
    {"id": 3, "name": "SE"},
]

LESION_TYPES = ["1. Microaneurysms", "2. Haemorrhages", "3. Hard Exudates", "4. Soft Exudates"]
CLASS_IDS = [0, 1, 2, 3]  # MA, HE, EX, SE

CATEGORIES_1CLS = [{"id": 0, "name": "lesion"}]


def _apply_min_box_size(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int, min_size: int = MIN_BOX_SIZE
) -> tuple[float, float, float, float]:
    """bbox 너비/높이가 min_size 미만이면 중심 고정 후 최소 크기로 확장. 이미지 경계 클리핑. (FGART와 동일)"""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(x2 - x1, min_size)
    h = max(y2 - y1, min_size)
    x1_new = max(0.0, cx - w / 2)
    y1_new = max(0.0, cy - h / 2)
    x2_new = min(float(img_w), cx + w / 2)
    y2_new = min(float(img_h), cy + h / 2)
    return x1_new, y1_new, x2_new, y2_new


def mask_to_bboxes_coco(
    mask_path: Path,
    min_area: int = MIN_AREA,
    min_box_size: int = MIN_BOX_SIZE,
) -> list[tuple[float, float, float, float]]:
    """IDRiD 마스크 (0/76 값) → connected components → COCO bbox [x, y, w, h]. 너무 작은 bbox는 min_box_size로 확장."""
    mask = np.array(Image.open(mask_path).convert("L"))
    H, W = mask.shape
    binary = (mask > 0).astype(np.uint8)  # IDRiD: 0=배경, 76=병변

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    bboxes = []
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x1 = float(stats[lbl, cv2.CC_STAT_LEFT])
        y1 = float(stats[lbl, cv2.CC_STAT_TOP])
        bw = float(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = float(stats[lbl, cv2.CC_STAT_HEIGHT])
        x2, y2 = x1 + bw, y1 + bh

        if min_box_size > 0:
            x1, y1, x2, y2 = _apply_min_box_size(x1, y1, x2, y2, W, H, min_box_size)
            bw = x2 - x1
            bh = y2 - y1

        bboxes.append((x1, y1, bw, bh))
    return bboxes


def collect_all_records(idrid_root: Path) -> list[dict]:
    """train + test 이미지 모두 수집, 각 이미지에 대해 lesion 마스크 경로 매칭"""
    seg_root = idrid_root / "A. Segmentation"
    img_dirs = {
        "train": seg_root / "1. Original Images" / "a. Training Set",
        "test": seg_root / "1. Original Images" / "b. Testing Set",
    }
    seg_dirs = {
        "train": seg_root / "2. All Segmentation Groundtruths" / "a. Training Set",
        "test": seg_root / "2. All Segmentation Groundtruths" / "b. Testing Set",
    }

    records = []
    seen_stems = set()

    for split, img_dir in img_dirs.items():
        if not img_dir.exists():
            continue
        seg_base = seg_dirs[split]
        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            stem = img_path.stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)

            masks = {}
            for les_name, cid in zip(LESION_TYPES, CLASS_IDS):
                les_dir = seg_base / les_name
                if not les_dir.exists():
                    continue
                cands = list(les_dir.glob(f"{stem}*.tif"))
                if cands:
                    masks[cid] = cands[0]

            records.append({
                "stem": stem,
                "img_path": img_path,
                "masks": masks,
            })

    return records


def build_coco_json(
    records: list[dict], out_root: Path, split_name: str = "test", is_1cls: bool = False
) -> Path:
    """COCO JSON 생성 + 이미지 symlink. is_1cls=True면 모든 bbox를 category_id=0으로."""
    img_dir = out_root / split_name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 0
    categories = CATEGORIES_1CLS if is_1cls else CATEGORIES_4CLS

    for image_id, rec in enumerate(records):
        img_p = Path(rec["img_path"])
        if not img_p.exists():
            continue

        with Image.open(img_p) as im:
            w, h = im.size

        file_name = img_p.name
        images.append({"id": image_id, "file_name": file_name, "width": w, "height": h})

        # symlink
        dst = img_dir / file_name
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(img_p.resolve())
        except OSError as e:
            print(f"[WARN] symlink failed: {img_p} -> {dst} ({e})")

        for cid, mask_p in rec["masks"].items():
            mask_p = Path(mask_p)
            if not mask_p.exists():
                continue
            cat_id = 0 if is_1cls else cid
            for x, y, bw, bh in mask_to_bboxes_coco(mask_p):
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    json_path = out_root / split_name / f"{split_name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    print(f"  {split_name}.json: images={len(images)}, annotations={len(annotations)}")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="IDRiD → COCO JSON (test only)")
    parser.add_argument("--idrid-root", type=Path, default=DEFAULT_IDRID_ROOT, help="IDRiD 원본 경로")
    parser.add_argument(
        "--mode",
        choices=["4cls", "1cls", "both"],
        default="both",
        help="4cls / 1cls / both",
    )
    args = parser.parse_args()

    if not args.idrid_root.exists():
        # 프로젝트 data 폴더 시도
        alt = Path(__file__).resolve().parents[2] / "data" / "IDRiD"
        if alt.exists():
            args.idrid_root = alt
            print(f"[INFO] Using project data: {alt}")
        else:
            print(f"[ERROR] IDRiD root not found: {args.idrid_root}")
            return 1

    print(f"[IDRiD] 레코드 구성 (root={args.idrid_root})...")
    records = collect_all_records(args.idrid_root)
    print(f"  전체 이미지: {len(records)}")

    print("\n[IDRiD] COCO JSON 생성 (test only)...")
    if args.mode in ("4cls", "both"):
        build_coco_json(records, OUT_4CLS, split_name="test", is_1cls=False)
        print(f"  → {OUT_4CLS}")
    if args.mode in ("1cls", "both"):
        build_coco_json(records, OUT_1CLS, split_name="test", is_1cls=True)
        print(f"  → {OUT_1CLS}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
