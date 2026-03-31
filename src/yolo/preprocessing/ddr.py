from pathlib import Path
import argparse
import json

BM_SPLITS = ("train", "valid", "test")
RE_SPLITS = ("train", "val", "test")
CROP_SPLITS = ("train", "valid", "test")

COCO_BM_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_4cls")
COCO_BM_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_1cls")
COCO_RE_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_4cls")
COCO_RE_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_1cls")
COCO_CROP_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_4cls")
COCO_CROP_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_1cls")

IMG_BM_ROOT = COCO_BM_4CLS_ROOT
IMG_RE_ROOT = COCO_RE_4CLS_ROOT
IMG_CROP_ROOT = COCO_CROP_4CLS_ROOT

YOLO_BM_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_yolo_4cls")
YOLO_BM_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_yolo_1cls")
YOLO_RE_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_yolo_4cls")
YOLO_RE_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_yolo_1cls")
YOLO_CROP_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_yolo_4cls")
YOLO_CROP_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_yolo_1cls")


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return cx, cy, wn, hn


def convert_coco_to_yolo(coco_root: Path, img_root: Path, yolo_root: Path, splits, is_1cls: bool = False):
    for split in splits:
        coco_json = coco_root / split / f"{split}.json"
        if not coco_json.exists():
            print(f"[WARN] COCO JSON not found: {coco_json}")
            continue

        with open(coco_json, "r", encoding="utf-8") as file:
            coco = json.load(file)

        images_by_id = {img["id"]: img for img in coco["images"]}
        anns_by_image = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            anns_by_image.setdefault(img_id, []).append(ann)

        src_img_dir = img_root / split / "images"
        if not src_img_dir.exists():
            print(f"[WARN] src_img_dir not found: {src_img_dir} (skip symlink)")
            src_img_dir = None

        img_out_dir = yolo_root / split / "images"
        lbl_out_dir = yolo_root / split / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{yolo_root.name}] {split}: images={len(images_by_id)}")

        for img_id, img_info in images_by_id.items():
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            src_img_path = src_img_dir / file_name if src_img_dir is not None else None
            dst_img_path = img_out_dir / file_name

            if src_img_path is not None and src_img_path.exists() and not dst_img_path.exists():
                try:
                    dst_img_path.symlink_to(src_img_path)
                except OSError as exc:
                    print(f"[WARN] symlink failed: {src_img_path} -> {dst_img_path} ({exc})")

            label_lines = []
            for ann in anns_by_image.get(img_id, []):
                class_id = 0 if is_1cls else int(ann["category_id"])
                cx, cy, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

            label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
            if label_lines:
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            else:
                label_path.write_text("", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bm", "re", "crop", "both"], default="both")
    parser.add_argument("--mode", choices=["4cls", "1cls", "both"], default="both")
    args = parser.parse_args()

    if args.dataset in ("bm", "both"):
        if args.mode in ("4cls", "both"):
            print("[YOLO] DDR_bm 4cls 변환 시작")
            convert_coco_to_yolo(COCO_BM_4CLS_ROOT, IMG_BM_ROOT, YOLO_BM_4CLS_ROOT, BM_SPLITS, is_1cls=False)
        if args.mode in ("1cls", "both"):
            print("[YOLO] DDR_bm 1cls 변환 시작")
            convert_coco_to_yolo(COCO_BM_1CLS_ROOT, IMG_BM_ROOT, YOLO_BM_1CLS_ROOT, BM_SPLITS, is_1cls=True)

    if args.dataset in ("re", "both"):
        if args.mode in ("4cls", "both"):
            print("[YOLO] DDR_re 4cls 변환 시작")
            convert_coco_to_yolo(COCO_RE_4CLS_ROOT, IMG_RE_ROOT, YOLO_RE_4CLS_ROOT, RE_SPLITS, is_1cls=False)
        if args.mode in ("1cls", "both"):
            print("[YOLO] DDR_re 1cls 변환 시작")
            convert_coco_to_yolo(COCO_RE_1CLS_ROOT, IMG_RE_ROOT, YOLO_RE_1CLS_ROOT, RE_SPLITS, is_1cls=True)

    if args.dataset in ("crop", "both"):
        if args.mode in ("4cls", "both"):
            print("[YOLO] DDR_crop 4cls 변환 시작")
            convert_coco_to_yolo(COCO_CROP_4CLS_ROOT, IMG_CROP_ROOT, YOLO_CROP_4CLS_ROOT, CROP_SPLITS, is_1cls=False)
        if args.mode in ("1cls", "both"):
            print("[YOLO] DDR_crop 1cls 변환 시작")
            convert_coco_to_yolo(COCO_CROP_1CLS_ROOT, IMG_CROP_ROOT, YOLO_CROP_1CLS_ROOT, CROP_SPLITS, is_1cls=True)

    print("Done.")


if __name__ == "__main__":
    main()
