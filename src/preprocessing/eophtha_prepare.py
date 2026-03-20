#!/usr/bin/env python3
"""
E-ophtha 원본 데이터 → COCO JSON 변환 스크립트.

전략:
- 4cls (DDR/FGART 통일): MA=0, HE=1, EX=2, SE=3  (E-ophtha에는 MA, EX만 존재)
- Negative 이미지 포함: EX healthy + MA healthy → empty annotations
- Stratified split: 클래스별(EX-only, MA-only, both, negative) 분포 고려, 환자 기준 7:1:2
- ex_healthy 포함 시 전체 434장

입력: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha/
  - e_optha_EX/{EX, healthy, Annotation_EX}
  - e_optha_MA/{MA, healthy, Annotation_MA}

출력 (COCO JSON + 이미지 symlink):
  - 4cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_4cls/{train,val,test}/{split}.json + images/
  - 1cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_1cls/{train,val,test}/{split}.json + images/
"""

from pathlib import Path
import json
import random
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# 경로
ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha")
EX_ROOT = ROOT / "e_optha_EX"
MA_ROOT = ROOT / "e_optha_MA"

EX_IMGS = EX_ROOT / "EX"
EX_HEALTHY = EX_ROOT / "healthy"
EX_MASKS = EX_ROOT / "Annotation_EX"

MA_IMGS = MA_ROOT / "MA"
MA_HEALTHY = MA_ROOT / "healthy"
MA_MASKS = MA_ROOT / "Annotation_MA"

# COCO JSON 출력
OUT_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_4cls")
OUT_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_1cls")

SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MIN_AREA = 25

# 4cls: MA=0, HE=1, EX=2, SE=3
CATEGORIES_4CLS = [
    {"id": 0, "name": "MA"},
    {"id": 1, "name": "HE"},
    {"id": 2, "name": "EX"},
    {"id": 3, "name": "SE"},
]
CLS_ID_MA = 0
CLS_ID_EX = 2


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix in IMG_EXTS and p.is_file()])


def patient_id(img_path: Path) -> str:
    """이미지 경로에서 E-number 환자 ID 추출 (e.g. E0000404)"""
    for part in img_path.parts:
        if part.startswith("E") and len(part) > 1 and part[1:].isdigit():
            return part
    return "UNKNOWN"


def mask_to_bboxes_coco(mask_path: Path, min_area: int = MIN_AREA) -> list[tuple[float, float, float, float]]:
    """바이너리 마스크 → connected components → COCO bbox [x, y, w, h].
    반환: [(x, y, w, h), ...]
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    H, W = mask.shape
    binary = (mask > 127).astype(np.uint8)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    bboxes = []
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x1 = stats[lbl, cv2.CC_STAT_LEFT]
        y1 = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        bboxes.append((float(x1), float(y1), float(bw), float(bh)))
    return bboxes


def build_records() -> pd.DataFrame:
    """전체 이미지 레코드 구성 (ex_healthy 포함 → 434장)"""
    ex_imgs = collect_images(EX_IMGS)
    ex_healthy = collect_images(EX_HEALTHY)
    ex_masks = collect_images(EX_MASKS)
    ma_imgs = collect_images(MA_IMGS)
    ma_healthy = collect_images(MA_HEALTHY)
    ma_masks = collect_images(MA_MASKS)

    ex_mask_stem_map = {p.stem.replace("_EX", ""): p for p in ex_masks}
    ex_img_stem_map = {p.stem: p for p in ex_imgs}
    ex_matched = {
        s: (ex_img_stem_map[s], ex_mask_stem_map[s])
        for s in ex_img_stem_map if s in ex_mask_stem_map
    }

    ma_mask_stem_map = {p.stem: p for p in ma_masks}
    ma_img_stem_map = {p.stem: p for p in ma_imgs}
    ma_matched = {
        s: (ma_img_stem_map[s], ma_mask_stem_map[s])
        for s in ma_img_stem_map if s in ma_mask_stem_map
    }

    records = {}

    for stem, (img_p, mask_p) in ex_matched.items():
        records[stem] = {
            "stem": stem,
            "img_path": img_p,
            "patient": patient_id(img_p),
            "ex_mask": mask_p,
            "ma_mask": None,
        }

    for stem, (img_p, mask_p) in ma_matched.items():
        if stem in records:
            records[stem]["ma_mask"] = mask_p
        else:
            records[stem] = {
                "stem": stem,
                "img_path": img_p,
                "patient": patient_id(img_p),
                "ex_mask": None,
                "ma_mask": mask_p,
            }

    for p in ma_healthy:
        stem = p.stem
        if stem not in records:
            records[stem] = {"stem": stem, "img_path": p, "patient": patient_id(p), "ex_mask": None, "ma_mask": None}

    for p in ex_healthy:
        stem = p.stem
        if stem not in records:
            records[stem] = {"stem": stem, "img_path": p, "patient": patient_id(p), "ex_mask": None, "ma_mask": None}

    return pd.DataFrame(records.values())


def stratified_split(records_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """클래스별(EX/MA) 분포를 고려한 patient-level stratified split (7:1:2).

    환자 stratum: ex_only, ma_only, both, negative
    - ex_only: EX만 있는 환자
    - ma_only: MA만 있는 환자
    - both: EX+MA 둘 다 있는 환자
    - negative: 병변 없는 환자
    """
    from collections import Counter
    from sklearn.model_selection import StratifiedShuffleSplit

    records_df = records_df.copy()
    records_df["has_ex"] = records_df["ex_mask"].notna()
    records_df["has_ma"] = records_df["ma_mask"].notna()

    # 환자별 stratum (클래스 조합)
    pt_has_ex = records_df.groupby("patient")["has_ex"].any()
    pt_has_ma = records_df.groupby("patient")["has_ma"].any()

    def patient_stratum(pid):
        ex = pt_has_ex.get(pid, False)
        ma = pt_has_ma.get(pid, False)
        if ex and ma:
            return "both"
        if ex:
            return "ex_only"
        if ma:
            return "ma_only"
        return "negative"

    patients = records_df["patient"].unique()
    strata = [patient_stratum(p) for p in patients]

    # 희소 stratum 병합 (2개 미만이면 인접 stratum에 합침)
    cnt = Counter(strata)
    merged = []
    for s in strata:
        if cnt[s] < 2:
            if s == "both":
                merged.append("ex_only" if cnt["ex_only"] >= cnt["ma_only"] else "ma_only")
            elif s == "ex_only" and cnt["ex_only"] < 2:
                merged.append("ma_only" if cnt["ma_only"] >= 2 else "lesion")
            elif s == "ma_only" and cnt["ma_only"] < 2:
                merged.append("ex_only" if cnt["ex_only"] >= 2 else "lesion")
            else:
                merged.append("lesion" if s in ("ex_only", "ma_only", "both") else s)
        else:
            merged.append(s)

    # train / val+test (70 vs 30)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, rest_idx = next(sss1.split(patients, merged))
    train_pts = set(patients[train_idx])
    rest_pts = patients[rest_idx]
    rest_strata = [merged[i] for i in rest_idx]

    # val+test → val / test (1:2 비율 → val 10%, test 20%)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=2 / 3, random_state=seed + 1)
    val_idx, test_idx = next(sss2.split(rest_pts, rest_strata))
    val_pts = set(rest_pts[val_idx])
    test_pts = set(rest_pts[test_idx])

    def assign_split(row):
        p = row["patient"]
        if p in test_pts:
            return "test"
        if p in val_pts:
            return "val"
        return "train"

    records_df["split"] = records_df.apply(assign_split, axis=1)
    return records_df


def _get_image_size(img_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as im:
        return im.size  # (w, h)


def build_coco_json(
    records_df: pd.DataFrame,
    out_root: Path,
    split_name: str,
    is_1cls: bool = False,
):
    """해당 split의 COCO JSON 생성 + 이미지 symlink"""
    subset = records_df[records_df["split"] == split_name]
    img_dir = out_root / split_name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 0

    for image_id, (_, row) in enumerate(subset.iterrows()):
        img_p = Path(row["img_path"])
        stem = row["stem"]
        if not img_p.exists():
            continue

        w, h = _get_image_size(img_p)
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

        # bbox from masks
        for mask_key, cat_id in [("ex_mask", CLS_ID_EX), ("ma_mask", CLS_ID_MA)]:
            mask_p = row.get(mask_key)
            if mask_p is None or (hasattr(mask_p, "__bool__") and not mask_p):
                continue
            mask_p = Path(mask_p)
            if not mask_p.exists():
                continue
            cid = 0 if is_1cls else cat_id
            for x, y, w, hh in mask_to_bboxes_coco(mask_p):
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [x, y, w, hh],
                    "area": w * hh,
                    "iscrowd": 0,
                })
                ann_id += 1

    categories = [{"id": 0, "name": "lesion"}] if is_1cls else CATEGORIES_4CLS
    coco = {"images": images, "annotations": annotations, "categories": categories}
    json_path = out_root / split_name / f"{split_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    print(f"  {split_name}.json: images={len(images)}, annotations={len(annotations)}")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="E-ophtha → COCO JSON")
    parser.add_argument("--mode", choices=["4cls", "1cls", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[E-ophtha] 레코드 구성...")
    records_df = build_records()
    records_df["has_ex"] = records_df["ex_mask"].notna()
    records_df["has_ma"] = records_df["ma_mask"].notna()
    records_df["negative"] = ~records_df["has_ex"] & ~records_df["has_ma"]

    print(f"  전체 이미지: {len(records_df)}")
    print(f"  has_ex: {records_df['has_ex'].sum()}, has_ma: {records_df['has_ma'].sum()}")
    print(f"  negative: {records_df['negative'].sum()}")
    print(f"  환자 수: {records_df['patient'].nunique()}")

    print("\n[E-ophtha] Stratified split (7:1:2, 클래스별 분포 고려)...")
    records_df = stratified_split(records_df, seed=args.seed)
    for sp in SPLITS:
        n = (records_df["split"] == sp).sum()
        sub = records_df[records_df["split"] == sp]
        ex_n = sub["has_ex"].sum()
        ma_n = sub["has_ma"].sum()
        neg_n = (~sub["has_ex"] & ~sub["has_ma"]).sum()
        print(f"  {sp}: {n} images (EX={ex_n}, MA={ma_n}, neg={neg_n})")

    if args.mode in ("4cls", "both"):
        print("\n[E-ophtha] 4cls COCO JSON 생성...")
        for sp in SPLITS:
            build_coco_json(records_df, OUT_4CLS, sp, is_1cls=False)
        print(f"  → {OUT_4CLS}")

    if args.mode in ("1cls", "both"):
        print("\n[E-ophtha] 1cls COCO JSON 생성...")
        for sp in SPLITS:
            build_coco_json(records_df, OUT_1CLS, sp, is_1cls=True)
        print(f"  → {OUT_1CLS}")

    print("Done.")


if __name__ == "__main__":
    main()
