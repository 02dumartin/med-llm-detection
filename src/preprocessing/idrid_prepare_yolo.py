#!/usr/bin/env python3
"""
IDRiD COCO JSON → YOLO 변환 스크립트.

입력 (idrid_prepare.py에서 생성):
  - 4cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_4cls/test/test.json
  - 1cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_1cls/test/test.json

출력:
  - 4cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_4cls/test/{images,labels}
  - 1cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_1cls/test/{images,labels}

이미지는 심볼릭 링크. YOLO 라벨: class_id cx cy w_norm h_norm (0-based, 0~1 정규화)
"""

from pathlib import Path
import json
import argparse

# COCO 입력 (idrid_prepare.py 출력)
COCO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_4cls")
COCO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_1cls")

# YOLO 출력
YOLO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_4cls")
YOLO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_1cls")

SPLIT = "test"


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO bbox [x, y, w, h] → YOLO [cx, cy, w_norm, h_norm]"""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return cx, cy, wn, hn


def convert_coco_to_yolo(coco_root: Path, yolo_root: Path, is_1cls: bool = False):
    """COCO test/test.json → YOLO test/{images,labels}"""
    coco_json = coco_root / SPLIT / f"{SPLIT}.json"
    if not coco_json.exists():
        print(f"[ERROR] COCO JSON not found: {coco_json}")
        return 1

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    src_img_dir = coco_root / SPLIT / "images"
    img_out_dir = yolo_root / SPLIT / "images"
    lbl_out_dir = yolo_root / SPLIT / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[IDRiD YOLO] {SPLIT}: images={len(images_by_id)}")

    for img_id, img_info in images_by_id.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img_path = src_img_dir / file_name
        dst_img_path = img_out_dir / file_name

        if src_img_path.exists():
            if dst_img_path.exists():
                dst_img_path.unlink()
            try:
                dst_img_path.symlink_to(src_img_path.resolve())
            except OSError as e:
                print(f"[WARN] symlink failed: {src_img_path} -> {dst_img_path} ({e})")

        anns = anns_by_image.get(img_id, [])
        label_lines = []
        for ann in anns:
            class_id = 0 if is_1cls else int(ann["category_id"])
            cx, cy, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

        label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
        if label_lines:
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        else:
            label_path.write_text("", encoding="utf-8")

    print(f"  → {yolo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="IDRiD COCO → YOLO 변환")
    parser.add_argument(
        "--mode",
        choices=["4cls", "1cls", "both"],
        default="both",
        help="4cls / 1cls / both",
    )
    args = parser.parse_args()

    ret = 0
    if args.mode in ("4cls", "both"):
        print("[IDRiD YOLO] 4cls 변환")
        ret |= convert_coco_to_yolo(COCO_4CLS_ROOT, YOLO_4CLS_ROOT, is_1cls=False)
    if args.mode in ("1cls", "both"):
        print("[IDRiD YOLO] 1cls 변환")
        ret |= convert_coco_to_yolo(COCO_1CLS_ROOT, YOLO_1CLS_ROOT, is_1cls=True)
    print("Done.")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
