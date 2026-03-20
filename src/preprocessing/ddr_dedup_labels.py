#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


DDR_ROOTS = [
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_1cls"),
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_4cls"),
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_1cls"),
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_4cls"),
]

YOLO_ROOTS = [
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_yolo_1cls"),
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_yolo_4cls"),
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_yolo_1cls"),
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_yolo_4cls"),
]

SPLITS = ["train", "valid", "val", "test"]


def dedup_coco_json(json_path: Path) -> int:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    anns = data.get("annotations", [])
    new_anns = []
    seen = set()
    removed = 0
    for ann in anns:
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if bbox is None:
            continue
        key = (img_id, cat_id, *bbox)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        new_anns.append(ann)
    if removed > 0:
        data["annotations"] = new_anns
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return removed


def dedup_yolo_labels(labels_dir: Path) -> int:
    removed = 0
    for p in labels_dir.glob("*.txt"):
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        uniq = list(dict.fromkeys(lines))
        if len(uniq) != len(lines):
            removed += len(lines) - len(uniq)
            p.write_text("\n".join(uniq) + "\n", encoding="utf-8")
    return removed


def main() -> None:
    total_removed = 0

    for root in DDR_ROOTS:
        if not root.exists():
            continue
        for split in SPLITS:
            json_path = root / split / f"{split}.json"
            if json_path.exists():
                removed = dedup_coco_json(json_path)
                if removed:
                    print(f"[COCO] {json_path} removed={removed}")
                total_removed += removed

    for root in YOLO_ROOTS:
        if not root.exists():
            continue
        for split in SPLITS:
            labels_dir = root / split / "labels"
            if labels_dir.exists():
                removed = dedup_yolo_labels(labels_dir)
                if removed:
                    print(f"[YOLO] {labels_dir} removed={removed}")
                total_removed += removed

    print(f"[DONE] total_removed={total_removed}")


if __name__ == "__main__":
    main()
