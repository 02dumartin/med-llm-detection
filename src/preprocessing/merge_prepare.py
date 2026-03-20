# src/preprocessing/merge_prepare.py

"""
FGART + DDR COCO 데이터 병합.

- train: FGART train + DDR train (파일명 충돌 방지: fgart__*, ddr__*)
- val: FGART val + DDR val
- test_fgart: FGART test만
- test_ddr: DDR test만

DDR 버전: DDR_4cls/DDR_1cls 사용. DDR는 valid 사용(val→valid 매핑).
4cls/1cls 모두 생성.
"""

from pathlib import Path
import json
import argparse

# 입력 COCO 루트
FGART_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_4scls")
FGART_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_1scls")
DDR_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_4cls")
DDR_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_1cls")

# 출력 루트
OUT_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_4scls")
OUT_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Merge_1scls")

MERGE_SPLITS = ("train", "val", "test_fgart", "test_ddr")
CATEGORIES_4CLS = [
    {"id": 0, "name": "MA"},
    {"id": 1, "name": "HE"},
    {"id": 2, "name": "EX"},
    {"id": 3, "name": "SE"},
]
CATEGORIES_1CLS = [{"id": 0, "name": "lesion"}]

FGART_PREFIX = "fgart__"
DDR_PREFIX = "ddr__"


def load_coco(coco_root: Path, split: str, use_valid: bool = False) -> dict | None:
    """COCO JSON 로드. split: train/val/test. use_valid: DDR용 val→valid 매핑"""
    actual_split = "valid" if (use_valid and split == "val") else split
    p = coco_root / actual_split / f"{actual_split}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_coco_split(
    fgart_coco: dict | None,
    ddr_coco: dict | None,
    out_split: str,
    is_train_or_val: bool,
) -> tuple[list, list]:
    """
    FGART + DDR COCO를 하나로 병합.
    is_train_or_val: True면 FGART+DDR 둘 다, False면 한 쪽만 (test_fgart 또는 test_ddr).
    Returns: (images, annotations)
    """
    images = []
    annotations = []
    next_img_id = 0
    next_ann_id = 0

    def add_dataset(coco: dict | None, prefix: str):
        nonlocal next_img_id, next_ann_id
        if coco is None:
            return
        old_to_new_img_id = {}
        for img in coco.get("images", []):
            new_name = f"{prefix}{img['file_name']}"
            images.append({
                "id": next_img_id,
                "file_name": new_name,
                "width": img["width"],
                "height": img["height"],
            })
            old_to_new_img_id[img["id"]] = next_img_id
            next_img_id += 1
        for ann in coco.get("annotations", []):
            new_img_id = old_to_new_img_id.get(ann["image_id"])
            if new_img_id is None:
                continue
            annotations.append({
                "id": next_ann_id,
                "image_id": new_img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0),
            })
            next_ann_id += 1

    if is_train_or_val:
        add_dataset(fgart_coco, FGART_PREFIX)
        add_dataset(ddr_coco, DDR_PREFIX)
    else:
        if "fgart" in out_split:
            add_dataset(fgart_coco, FGART_PREFIX)
        else:
            add_dataset(ddr_coco, DDR_PREFIX)

    return images, annotations


def link_image(src_dir: Path, file_name: str, dst_path: Path):
    """src_dir/file_name을 dst_path에 심볼릭 링크."""
    src = src_dir / file_name
    if not src.exists():
        return False
    if dst_path.exists():
        dst_path.unlink()
    try:
        dst_path.symlink_to(src.resolve())
        return True
    except OSError as e:
        print(f"[WARN] link failed {src} -> {dst_path}: {e}")
        return False


def run_merge(is_4cls: bool = True, is_1cls: bool = True):
    """FGART + DDR 병합 실행."""
    out_root = OUT_4CLS if is_4cls else OUT_1CLS
    fgart_root = FGART_4CLS if is_4cls else FGART_1CLS
    ddr_root = DDR_4CLS if is_4cls else DDR_1CLS
    categories = CATEGORIES_4CLS if is_4cls else CATEGORIES_1CLS

    label = "4cls" if is_4cls else "1cls"
    print(f"\n[Merge {label}]")

    for out_split in MERGE_SPLITS:
        is_train_or_val = out_split in ("train", "val")
        if is_train_or_val:
            fgart_split = out_split
            ddr_split = out_split
        else:
            fgart_split = "test" if "fgart" in out_split else None
            ddr_split = "test" if "ddr" in out_split else None

        fgart_coco = load_coco(fgart_root, fgart_split) if fgart_split else None
        ddr_coco = load_coco(ddr_root, ddr_split, use_valid=True) if ddr_split else None

        if not is_train_or_val and fgart_coco is None and ddr_coco is None:
            continue

        images, annotations = merge_coco_split(
            fgart_coco, ddr_coco, out_split, is_train_or_val
        )
        if not images:
            print(f"  {out_split}: (empty, skip)")
            continue

        # COCO JSON 저장
        (out_root / out_split).mkdir(parents=True, exist_ok=True)
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        json_path = out_root / out_split / f"{out_split}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(f"  {out_split}: images={len(images)}, annotations={len(annotations)}")

        # 이미지 심볼릭 링크
        img_dir = out_root / out_split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        def link_from_coco(coco: dict | None, src_root: Path, prefix: str, use_valid: bool = False):
            if coco is None:
                return
            split_name = "train" if out_split == "train" else "val" if out_split == "val" else "test"
            if use_valid and split_name == "val":
                split_name = "valid"  # DDR는 valid 사용
            src_img_dir = src_root / split_name / "images"
            for img in coco["images"]:
                orig_name = img["file_name"]
                new_name = f"{prefix}{orig_name}"
                dst = img_dir / new_name
                link_image(src_img_dir, orig_name, dst)

        if is_train_or_val:
            link_from_coco(fgart_coco, fgart_root, FGART_PREFIX, use_valid=False)
            link_from_coco(ddr_coco, ddr_root, DDR_PREFIX, use_valid=True)
        else:
            if "fgart" in out_split and fgart_coco:
                link_from_coco(fgart_coco, fgart_root, FGART_PREFIX, use_valid=False)
            elif "ddr" in out_split and ddr_coco:
                link_from_coco(ddr_coco, ddr_root, DDR_PREFIX, use_valid=True)


def main():
    parser = argparse.ArgumentParser(description="FGART + DDR COCO merge")
    parser.add_argument(
        "--mode",
        choices=["4cls", "1cls", "both"],
        default="both",
    )
    args = parser.parse_args()

    print("[Merge] FGART + DDR → Merge_4scls / Merge_1scls")
    print("  splits: train, val, test_fgart, test_ddr")

    if args.mode in ("4cls", "both"):
        run_merge(is_4cls=True, is_1cls=False)
    if args.mode in ("1cls", "both"):
        run_merge(is_4cls=False, is_1cls=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
