# src/preprocessing/merge_prepare_yolo.py

"""
Merge (FGART + DDR / DDR_crop) COCO → YOLO 변환.

입력 (merge_prepare.py 실행 후):
  - raw 4cls:  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_4scls/{train,val,test_fgart,test_ddr}/{split}.json
  - raw 1cls:  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_1scls/{train,val,test_fgart,test_ddr}/{split}.json
  - crop 4cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_crop_4scls/{train,val,test_fgart,test_ddr}/{split}.json
  - crop 1cls: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_crop_1scls/{train,val,test_fgart,test_ddr}/{split}.json

출력:
  - raw 4cls YOLO:  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_yolo_4cls/{train,val,test_fgart,test_ddr}/{images,labels}
  - raw 1cls YOLO:  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_yolo_1cls/{train,val,test_fgart,test_ddr}/{images,labels}
  - crop 4cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_crop_yolo_4cls/{train,val,test_fgart,test_ddr}/{images,labels}
  - crop 1cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_crop_yolo_1cls/{train,val,test_fgart,test_ddr}/{images,labels}

이미지는 심볼릭 링크. 파일명은 fgart__*, ddr__* prefix 유지.
"""

from pathlib import Path
import json
import argparse

MERGE_SPLITS = ("train", "val", "test_fgart", "test_ddr")

RAW_COCO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_4scls")
RAW_COCO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_1scls")
RAW_YOLO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_yolo_4cls")
RAW_YOLO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_yolo_1cls")

CROP_COCO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_crop_4scls")
CROP_COCO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_crop_1scls")
CROP_YOLO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_crop_yolo_4cls")
CROP_YOLO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_crop_yolo_1cls")


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO bbox [x, y, w, h] → YOLO [cx, cy, w_norm, h_norm]"""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return cx, cy, wn, hn


def convert_coco_to_yolo(
    coco_root: Path,
    yolo_root: Path,
    splits: tuple[str, ...],
    is_1cls: bool = False,
):
    """COCO JSON → YOLO labels + 이미지 심볼릭 링크."""
    for split in splits:
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

            if dst_img_path.is_symlink() or dst_img_path.exists():
                dst_img_path.unlink()
            if src_img_path.exists():
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


def get_variant_roots(ddr_source: str, is_1cls: bool) -> tuple[Path, Path]:
    if ddr_source == "crop":
        return (
            CROP_COCO_1CLS_ROOT if is_1cls else CROP_COCO_4CLS_ROOT,
            CROP_YOLO_1CLS_ROOT if is_1cls else CROP_YOLO_4CLS_ROOT,
        )
    return (
        RAW_COCO_1CLS_ROOT if is_1cls else RAW_COCO_4CLS_ROOT,
        RAW_YOLO_1CLS_ROOT if is_1cls else RAW_YOLO_4CLS_ROOT,
    )


def main():
    parser = argparse.ArgumentParser(description="Merge COCO → YOLO 변환")
    parser.add_argument(
        "--ddr-source",
        choices=["raw", "crop"],
        default="crop",
        help="어떤 DDR 버전으로 만든 merge COCO를 YOLO로 변환할지 선택",
    )
    parser.add_argument(
        "--mode",
        choices=["4cls", "1cls", "both"],
        default="both",
    )
    args = parser.parse_args()

    if args.ddr_source == "crop":
        print("[Merge YOLO] Merge_crop_4scls / Merge_crop_1scls → Merge_crop_yolo_*")
    else:
        print("[Merge YOLO] Merge_4scls / Merge_1scls → Merge_yolo_*")
    print("  splits: train, val, test_fgart, test_ddr")

    if args.mode in ("4cls", "both"):
        coco_root, yolo_root = get_variant_roots(args.ddr_source, is_1cls=False)
        print("\n[YOLO] 4cls 변환")
        convert_coco_to_yolo(
            coco_root=coco_root,
            yolo_root=yolo_root,
            splits=MERGE_SPLITS,
            is_1cls=False,
        )
    if args.mode in ("1cls", "both"):
        coco_root, yolo_root = get_variant_roots(args.ddr_source, is_1cls=True)
        print("\n[YOLO] 1cls 변환")
        convert_coco_to_yolo(
            coco_root=coco_root,
            yolo_root=yolo_root,
            splits=MERGE_SPLITS,
            is_1cls=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
