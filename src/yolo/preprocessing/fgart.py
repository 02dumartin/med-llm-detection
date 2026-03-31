from pathlib import Path
import argparse
import json

SPLITS = ("train", "val", "test")

COCO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_4scls")
COCO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_1scls")

YOLO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_yolo_4cls")
YOLO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_yolo_1cls")


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return cx, cy, wn, hn


def convert_coco_to_yolo(coco_root: Path, yolo_root: Path, is_1cls: bool = False):
    for split in SPLITS:
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
            try:
                dst_img_path.symlink_to(src_img_path)
            except OSError as exc:
                print(f"[WARN] symlink failed: {src_img_path} -> {dst_img_path} ({exc})")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["4cls", "1cls", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("4cls", "both"):
        print("[YOLO] 4cls 변환 시작")
        convert_coco_to_yolo(COCO_4CLS_ROOT, YOLO_4CLS_ROOT, is_1cls=False)

    if args.mode in ("1cls", "both"):
        print("[YOLO] 1cls 변환 시작")
        convert_coco_to_yolo(COCO_1CLS_ROOT, YOLO_1CLS_ROOT, is_1cls=True)

    print("Done.")


if __name__ == "__main__":
    main()
