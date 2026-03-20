#!/usr/bin/env python3
"""
E-ophtha COCO JSON → YOLO 변환 스크립트.

입력 (eophtha_prepare.py에서 생성):
  - 4cls COCO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_4cls/{train,val,test}/{split}.json
  - 1cls COCO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_1cls/{train,val,test}/{split}.json

출력:
  - 4cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/E-OPTHA_yolo_4cls/{train,val,test}/{images,labels}
  - 1cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/E-OPTHA_yolo_1cls/{train,val,test}/{images,labels}

이미지는 심볼릭 링크로 연결.
YOLO 라벨 형식: class_id cx cy w_norm h_norm  (0-based, 0~1 정규화)
"""

from pathlib import Path
import json
import argparse

SPLITS = ("train", "val", "test")

# COCO 입력 루트 (eophtha_prepare.py 출력)
COCO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_4cls")
COCO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_1cls")

# YOLO 출력 루트
YOLO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/E-OPTHA_yolo_4cls")
YOLO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/E-OPTHA_yolo_1cls")


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO bbox [x, y, w, h] → YOLO [cx, cy, w_norm, h_norm]"""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return cx, cy, wn, hn


def convert_coco_to_yolo(coco_root: Path, yolo_root: Path, is_1cls: bool = False):
    """
    coco_root/<split>/<split>.json 읽어서
    yolo_root/<split>/{images,labels} 생성.
    """
    for split in SPLITS:
        coco_json = coco_root / split / f"{split}.json"
        if not coco_json.exists():
            print(f"[WARN] COCO JSON not found: {coco_json}")
            continue

        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images_by_id = {img["id"]: img for img in coco["images"]}

        anns_by_image = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            anns_by_image.setdefault(img_id, []).append(ann)

        src_img_dir = coco_root / split / "images"
        img_out_dir = yolo_root / split / "images"
        lbl_out_dir = yolo_root / split / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{yolo_root.name}] {split}: images={len(images_by_id)}")

        for img_id, img_info in images_by_id.items():
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            src_img_path = src_img_dir / file_name
            dst_img_path = img_out_dir / file_name

            if dst_img_path.exists():
                dst_img_path.unlink()
            try:
                dst_img_path.symlink_to(src_img_path.resolve())
            except OSError as e:
                print(f"[WARN] symlink failed: {src_img_path} -> {dst_img_path} ({e})")

            anns = anns_by_image.get(img_id, [])
            label_lines = []
            for ann in anns:
                if is_1cls:
                    class_id = 0
                else:
                    class_id = int(ann["category_id"])

                cx, cy, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

            label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
            if label_lines:
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            else:
                label_path.write_text("", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E-ophtha COCO → YOLO 변환")
    parser.add_argument(
        "--mode",
        choices=["4cls", "1cls", "both"],
        default="both",
        help="4cls / 1cls / both",
    )
    args = parser.parse_args()

    if args.mode in ("4cls", "both"):
        print("[E-ophtha YOLO] 4cls 변환")
        convert_coco_to_yolo(
            coco_root=COCO_4CLS_ROOT,
            yolo_root=YOLO_4CLS_ROOT,
            is_1cls=False,
        )

    if args.mode in ("1cls", "both"):
        print("[E-ophtha YOLO] 1cls 변환")
        convert_coco_to_yolo(
            coco_root=COCO_1CLS_ROOT,
            yolo_root=YOLO_1CLS_ROOT,
            is_1cls=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()
