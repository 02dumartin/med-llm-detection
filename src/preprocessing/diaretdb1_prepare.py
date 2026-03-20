#!/usr/bin/env python3
"""
DIARETDB1 원본 데이터 → COCO JSON 변환 스크립트.

- IDRiD와 동일: train + test 모두 합쳐 test로만 출력 (구분 없음)
- 4cls: MA=0, HE=1, EX=2, SE=3 (0-based, Disc 제외)
- 1cls: lesion=0
- XML 파싱 (centroid+radius → bbox)
- 해상도 고려 min_box_size: max(16, min_dim * 0.02)

입력: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DIARETDB1/ddb1_v02_01
  - images/
  - groundtruth/
  - ddb1_v02_01_train.txt, ddb1_v02_01_test.txt

출력:
  - 4cls: DIARETDB1_4cls/test/test.json + images/
  - 1cls: DIARETDB1_1cls/test/test.json + images/
"""

from pathlib import Path
import json
import argparse
import xml.etree.ElementTree as ET

from PIL import Image

# 경로
DEFAULT_DIARETDB1_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DIARETDB1/ddb1_v02_01")
OUT_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DIARETDB1_4cls")
OUT_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DIARETDB1_1cls")

# 4cls: MA=0, HE=1, EX=2, SE=3 (DDR/FGART 통일)
CATEGORIES_4CLS = [
    {"id": 0, "name": "MA"},
    {"id": 1, "name": "HE"},
    {"id": 2, "name": "EX"},
    {"id": 3, "name": "SE"},
]
CATEGORIES_1CLS = [{"id": 0, "name": "lesion"}]

# XML markingtype → 4cls id (Disc 제외)
MARKING_TO_ID = {
    "Red_small_dots": 0,  # MA
    "Haemorrhages": 1,
    "Hard_exudates": 2,
    "Soft_exudates": 3,
    "Disc": None,
}


def _apply_min_box_size(x1: float, y1: float, x2: float, y2: float,
                        img_w: int, img_h: int, min_size: int) -> tuple[float, float, float, float]:
    """bbox 너비/높이가 min_size 미만이면 중심 고정 후 확장. 이미지 경계 클리핑."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(x2 - x1, min_size)
    h = max(y2 - y1, min_size)
    x1_new = max(0.0, cx - w / 2)
    y1_new = max(0.0, cy - h / 2)
    x2_new = min(float(img_w), cx + w / 2)
    y2_new = min(float(img_h), cy + h / 2)
    return x1_new, y1_new, x2_new, y2_new


def _get_min_box_size(img_w: int, img_h: int) -> int:
    """해상도 고려 최소 bbox 크기 (짧은 변의 2%, 최소 16)"""
    min_dim = min(img_w, img_h)
    return max(16, int(min_dim * 0.02))


def parse_xml_to_bboxes(xml_path: Path, img_w: int, img_h: int) -> list[tuple[int, float, float, float, float]]:
    """XML 파싱 → [(cat_id, x, y, w, h), ...] COCO 형식. Disc 제외."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    min_size = _get_min_box_size(img_w, img_h)
    bboxes = []
    for m in root.findall(".//marking"):
        mtype = m.find("markingtype")
        if mtype is None or mtype.text is None:
            continue
        cat_id = MARKING_TO_ID.get(mtype.text.strip())
        if cat_id is None:
            continue
        circ = m.find(".//circleregion")
        ell = m.find(".//ellipseregion")
        cx, cy = 0.0, 0.0
        rx, ry = 5.0, 5.0
        if circ is not None:
            cent = circ.find("centroid/coords2d")
            r = circ.find("radius[@direction='x']")
            if cent is not None and r is not None:
                cx, cy = map(int, cent.text.split(","))
                rx = int(r.text) if r.text else 5
                ry = rx
        elif ell is not None:
            cent = ell.find("centroid/coords2d")
            rx_el = ell.find("radius[@direction='x']")
            ry_el = ell.find("radius[@direction='y']")
            if cent is not None:
                cx, cy = map(int, cent.text.split(","))
            if rx_el is not None and rx_el.text:
                rx = int(rx_el.text)
            if ry_el is not None and ry_el.text:
                ry = int(ry_el.text)
        rx, ry = max(rx, 2), max(ry, 2)
        x1, y1 = float(cx - rx), float(cy - ry)
        x2, y2 = float(cx + rx), float(cy + ry)
        if min_size > 0:
            x1, y1, x2, y2 = _apply_min_box_size(x1, y1, x2, y2, img_w, img_h, min_size)
        w = x2 - x1
        h = y2 - y1
        bboxes.append((cat_id, x1, y1, w, h))
    return bboxes


def _parse_split_txt(root: Path, txt_path: Path) -> list[dict]:
    """split txt 파싱 → [(img_path, [xml_paths]), ...]"""
    if not txt_path.exists():
        return []
    records = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            img_rel = parts[0]
            img_path = root / img_rel
            if not img_path.exists():
                img_path = root / img_rel.replace(".png", ".ppm")
            if not img_path.exists():
                continue
            xml_paths = []
            for part in parts[1:]:
                xml_path = root / part if part.startswith("groundtruth/") else root / "groundtruth" / part
                if xml_path.exists():
                    xml_paths.append(xml_path)
            if not xml_paths:
                xml_paths = list((root / "groundtruth").glob(f"{img_path.stem}_*.xml"))
            records.append({"img_path": img_path, "xml_paths": xml_paths})
    return records


def collect_test_records(root: Path) -> list[dict]:
    """ddb1_v02_01_test.txt 파싱"""
    return _parse_split_txt(root, root / "ddb1_v02_01_test.txt")


def collect_train_records(root: Path) -> list[dict]:
    """ddb1_v02_01_train.txt 파싱"""
    return _parse_split_txt(root, root / "ddb1_v02_01_train.txt")


def build_coco_json(records: list[dict], out_root: Path, split_name: str = "test", is_1cls: bool = False):
    """COCO JSON 생성 + 이미지 symlink."""
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
            img_w, img_h = im.size

        file_name = img_p.name
        images.append({"id": image_id, "file_name": file_name, "width": img_w, "height": img_h})

        dst = img_dir / file_name
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(img_p.resolve())
        except OSError as e:
            print(f"[WARN] symlink failed: {img_p} -> {dst} ({e})")

        seen = set()
        for xml_p in rec["xml_paths"]:
            for cat_id, x, y, w, h in parse_xml_to_bboxes(xml_p, img_w, img_h):
                out_cat = 0 if is_1cls else cat_id
                key = (out_cat, round(x, 1), round(y, 1))
                if key in seen:
                    continue
                seen.add(key)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": out_cat,
                    "bbox": [x, y, w, h],
                    "area": w * h,
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
    parser = argparse.ArgumentParser(description="DIARETDB1 → COCO JSON (train+test 합쳐 test로, IDRiD와 동일)")
    parser.add_argument("--root", type=Path, default=DEFAULT_DIARETDB1_ROOT, help="ddb1_v02_01 경로")
    parser.add_argument("--mode", choices=["4cls", "1cls", "both"], default="both")
    args = parser.parse_args()

    if not args.root.exists():
        alt = Path(__file__).resolve().parents[2] / "data" / "DIARETDB1" / "ddb1_v02_01"
        if alt.exists():
            args.root = alt
            print(f"[INFO] Using: {alt}")
        else:
            print(f"[ERROR] Root not found: {args.root}")
            return 1

    print(f"[DIARETDB1] 레코드 구성 (root={args.root})...")
    train_records = collect_train_records(args.root)
    test_records = collect_test_records(args.root)
    all_records = train_records + test_records
    print(f"  train: {len(train_records)}, test: {len(test_records)} → 전체 {len(all_records)}장을 test로 출력")

    print("\n[DIARETDB1] COCO JSON 생성 (train+test 합쳐 test로)...")
    if args.mode in ("4cls", "both"):
        build_coco_json(all_records, OUT_4CLS, split_name="test", is_1cls=False)
        print(f"  → {OUT_4CLS}")
    if args.mode in ("1cls", "both"):
        build_coco_json(all_records, OUT_1CLS, split_name="test", is_1cls=True)
        print(f"  → {OUT_1CLS}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
