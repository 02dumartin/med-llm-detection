"""
DDR crop 데이터 준비:

1) lesion_segmentation 원본 이미지를 train/valid/test 모두 fundus 영역만 crop
2) crop 기준으로 detection XML bbox를 remap
3) COCO 4cls / 1cls JSON 생성

출력:
  - /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_4cls
  - /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_1cls
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


DDR_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR")
LESION_DET_ROOT = DDR_ROOT / "lesion_detection"
SEG_ROOT = DDR_ROOT / "lesion_segmentation"

OUT_DDR_CROP_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_4cls")
OUT_DDR_CROP_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_1cls")

SEG_IMG_DIRS = {
    "train": SEG_ROOT / "train" / "image",
    "valid": SEG_ROOT / "valid" / "image",
    "test": SEG_ROOT / "test" / "image",
}

SPLITS = ("train", "valid", "test")

CLASS_NAME_TO_ID = {
    "MA": 0,
    "HE": 1,
    "EX": 2,
    "SE": 3,
}

XML_CLASS_TO_NAME = {
    "ma": "MA",
    "he": "HE",
    "ex": "EX",
    "se": "SE",
}

FUNDUS_THRESHOLD = 8
PAD_RATIO = 0.02
MIN_PAD = 10
MIN_BOX_SIDE = 1


def create_split_dirs(out_root: Path, splits) -> None:
    for split_name in splits:
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)


def _get_image_size(img_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as im:
        return im.size


def parse_voc_xml(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    boxes = []
    seen = set()
    for obj in tree.findall("object"):
        raw_name = obj.find("name").text
        if raw_name is None:
            continue
        full_name = XML_CLASS_TO_NAME.get(raw_name.strip().lower())
        if full_name is None:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
        except (TypeError, ValueError):
            continue

        key = (full_name, xmin, ymin, xmax, ymax)
        if key in seen:
            continue
        seen.add(key)
        boxes.append(
            {
                "class_name": full_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )
    return boxes


def find_seg_image_for_xml(xml_path: Path, seg_img_dirs=SEG_IMG_DIRS):
    stem = xml_path.stem
    for split_name, img_dir in seg_img_dirs.items():
        if not img_dir.exists():
            continue
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            img_path = img_dir / f"{stem}{ext}"
            if img_path.exists():
                return img_path, split_name
    return None, None


def load_pairs_for_ddr_split(ann_split: str) -> list[dict]:
    ann_dir = LESION_DET_ROOT / ann_split
    pairs = []
    for xml_path in sorted(ann_dir.glob("*.xml")):
        img_path, seg_split = find_seg_image_for_xml(xml_path)
        if img_path is None:
            continue
        pairs.append(
            {
                "image": img_path,
                "xml": xml_path,
                "ann_split": ann_split,
                "seg_split": seg_split,
            }
        )
    return pairs


def compute_fundus_crop_box(img_path: Path) -> tuple[int, int, int, int]:
    with Image.open(img_path) as im:
        rgb = np.asarray(im.convert("RGB"))

    mask = rgb.max(axis=2) > FUNDUS_THRESHOLD
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    height, width = mask.shape

    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, width, height)

    y1 = int(rows[0])
    y2 = int(rows[-1]) + 1
    x1 = int(cols[0])
    x2 = int(cols[-1]) + 1

    pad_x = max(MIN_PAD, int((x2 - x1) * PAD_RATIO))
    pad_y = max(MIN_PAD, int((y2 - y1) * PAD_RATIO))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return (x1, y1, x2, y2)


def remap_box_to_crop(box: dict, crop_box: tuple[int, int, int, int]) -> dict | None:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    x1 = max(box["xmin"], crop_x1) - crop_x1
    y1 = max(box["ymin"], crop_y1) - crop_y1
    x2 = min(box["xmax"], crop_x2) - crop_x1
    y2 = min(box["ymax"], crop_y2) - crop_y1

    if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
        return None

    return {
        "class_name": box["class_name"],
        "xmin": int(x1),
        "ymin": int(y1),
        "xmax": int(x2),
        "ymax": int(y2),
    }


def prepare_cropped_images(split_to_pairs: dict[str, list[dict]]) -> dict[str, list[dict]]:
    create_split_dirs(OUT_DDR_CROP_4CLS, SPLITS)
    create_split_dirs(OUT_DDR_CROP_1CLS, SPLITS)

    processed: dict[str, list[dict]] = {split_name: [] for split_name in SPLITS}

    for split_name in SPLITS:
        for pair in split_to_pairs.get(split_name, []):
            img_path = pair["image"]
            crop_box = compute_fundus_crop_box(img_path)
            out_img_4cls = OUT_DDR_CROP_4CLS / split_name / "images" / img_path.name
            out_img_1cls = OUT_DDR_CROP_1CLS / split_name / "images" / img_path.name

            with Image.open(img_path) as im:
                cropped = im.crop(crop_box)
                cropped.save(out_img_4cls)

            if out_img_1cls.exists() or out_img_1cls.is_symlink():
                out_img_1cls.unlink()
            try:
                out_img_1cls.symlink_to(out_img_4cls)
            except OSError:
                with Image.open(out_img_4cls) as im:
                    im.save(out_img_1cls)

            processed[split_name].append(
                {
                    **pair,
                    "crop_box": crop_box,
                    "image": out_img_4cls,
                }
            )
    return processed


def build_coco_4cls(out_root: Path, split_to_pairs: dict[str, list[dict]], splits) -> None:
    categories_4cls = [
        {"id": 0, "name": "MA"},
        {"id": 1, "name": "HE"},
        {"id": 2, "name": "EX"},
        {"id": 3, "name": "SE"},
    ]

    for split_name in splits:
        pairs = split_to_pairs.get(split_name, [])
        images = []
        annotations = []
        ann_id = 0

        for image_id, pair in enumerate(pairs):
            img_path = pair["image"]
            xml_path = pair["xml"]
            crop_box = pair["crop_box"]
            width, height = _get_image_size(img_path)

            images.append(
                {
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                }
            )

            for box in parse_voc_xml(xml_path):
                remapped = remap_box_to_crop(box, crop_box)
                if remapped is None:
                    continue
                cid = CLASS_NAME_TO_ID[remapped["class_name"]]
                x1 = remapped["xmin"]
                y1 = remapped["ymin"]
                x2 = remapped["xmax"]
                y2 = remapped["ymax"]
                bw = x2 - x1
                bh = y2 - y1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cid,
                        "bbox": [float(x1), float(y1), float(bw), float(bh)],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        (out_root / split_name).mkdir(parents=True, exist_ok=True)
        json_path = out_root / split_name / f"{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": categories_4cls,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(
            f"[OK] 4cls COCO ({out_root.name}) {split_name}: "
            f"images={len(images)}, annotations={len(annotations)}"
        )


def build_coco_1cls_from_4cls(out_root_4cls: Path, out_root_1cls: Path, splits) -> None:
    categories_1cls = [{"id": 0, "name": "lesion"}]
    for split_name in splits:
        in_path = out_root_4cls / split_name / f"{split_name}.json"
        with open(in_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        for ann in coco.get("annotations", []):
            ann["category_id"] = 0
        coco["categories"] = categories_1cls

        (out_root_1cls / split_name).mkdir(parents=True, exist_ok=True)
        out_path = out_root_1cls / split_name / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(
            f"[OK] 1cls COCO ({out_root_1cls.name}) {split_name}: "
            f"images={len(coco['images'])}, annotations={len(coco['annotations'])}"
        )


def print_split_summary(split_to_pairs: dict[str, list[dict]]) -> None:
    print("\n[Summary] DDR_crop split별 이미지/레이즌 통계")
    total_imgs = sum(len(v) for v in split_to_pairs.values())
    for split_name in SPLITS:
        pairs = split_to_pairs.get(split_name, [])
        n_imgs = len(pairs)
        ratio = (n_imgs / total_imgs * 100.0) if total_imgs > 0 else 0.0
        n_instances = 0
        for pair in pairs:
            for box in parse_voc_xml(pair["xml"]):
                if remap_box_to_crop(box, pair["crop_box"]) is not None:
                    n_instances += 1
        print(f"  {split_name}: images={n_imgs} ({ratio:.1f}%), instances={n_instances}")


def main() -> None:
    print("[DDR_crop] load_pairs_for_ddr_split()")
    split_to_pairs = {split_name: load_pairs_for_ddr_split(split_name) for split_name in SPLITS}
    for split_name, pairs in split_to_pairs.items():
        print(f"  {split_name}: {len(pairs)}")

    print("[DDR_crop] prepare_cropped_images()")
    cropped_pairs = prepare_cropped_images(split_to_pairs)

    print("[DDR_crop] build_coco_4cls()")
    build_coco_4cls(OUT_DDR_CROP_4CLS, cropped_pairs, SPLITS)

    print("[DDR_crop] build_coco_1cls_from_4cls()")
    build_coco_1cls_from_4cls(OUT_DDR_CROP_4CLS, OUT_DDR_CROP_1CLS, SPLITS)

    print_split_summary(cropped_pairs)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
