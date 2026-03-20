# src/preprocessing/ddr_prepare.py

"""
DDR 데이터 준비:

1) DDR_bm (benchmark: 기존 train/valid/test split 그대로 사용)
   - lesion_detection/{train,valid,test} XML + segmentation image 기준
   - COCO 4cls JSON (MA/HE/EX/SE, id=0,1,2,3)
   - COCO 1cls JSON (lesion, id=0)

2) DDR_re (re-split: 전체 757장을 train:test:val = 7:2:1로 stratified split)
   - 전체 XML + segmentation image 기준
   - COCO 4cls / COCO 1cls

FGART와 동일하게:
- 4cls categories: MA=0, HE=1, EX=2, SE=3
- 1cls: lesion=0
"""

from pathlib import Path
import json
import sys
from typing import Dict, List

from sklearn.model_selection import StratifiedShuffleSplit

# --- 경로 / 상수 ---

# 원본 입력 / COCO+이미지 출력: 모두 데이터 루트 하위 (FGART_4scls와 동일 구조)
DDR_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR")

LESION_DET_ROOT = DDR_ROOT / "lesion_detection"       # train/valid/test 안에 xml
SEG_ROOT = DDR_ROOT / "lesion_segmentation"           # train/valid/test/image 안에 jpg

# benchmark용 COCO+이미지 출력 (데이터 루트 하위)
OUT_DDR_BM_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_4cls")
OUT_DDR_BM_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_1cls")

# re-split용 COCO+이미지 출력 (데이터 루트 하위)
OUT_DDR_RE_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_4cls")
OUT_DDR_RE_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_1cls")

# segmentation 이미지 디렉토리 (노트북 구조 그대로)
SEG_IMG_DIRS = {
    "train": SEG_ROOT / "train" / "image",
    "valid": SEG_ROOT / "valid" / "image",
    "test": SEG_ROOT / "test" / "image",
}

# benchmark split 이름 (DDR는 valid 그대로)
BM_SPLITS = ("train", "valid", "test")

# re-split 이름 (FGART 스타일로 val 사용)
RE_SPLITS = ("train", "val", "test")
RE_SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train:test:val = 7:2:1

DEFAULT_SEED = 42

# 4cls: MA=0, HE=1, EX=2, SE=3 (FGART와 완전 통일, 대문자)
CLASS_NAME_TO_ID = {
    "MA": 0,
    "HE": 1,
    "EX": 2,
    "SE": 3,
}

# XML에는 소문자라서, 소문자→대문자 매핑
XML_CLASS_TO_NAME = {
    "ma": "MA",
    "he": "HE",
    "ex": "EX",
    "se": "SE",
}


# --- 유틸 함수 ---

def _get_image_size(img_path):
    """이미지 (width, height). PIL 사용."""
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size  # (w, h)


def create_split_dirs(out_root: Path, splits):
    """
    out_root/<split>/images 디렉터리 생성.
    FGART 준비 스크립트와 동일한 구조를 맞추기 위함.
    """
    for s in splits:
        (out_root / s / "images").mkdir(parents=True, exist_ok=True)


def link_images_per_split(out_root: Path, split_to_pairs, splits):
    """
    각 split별 이미지들을 out_root/<split>/images/ 에 심볼릭 링크로 생성.
    DDR에서는 segmentation 이미지 원본을 링크 대상으로 사용.
    """
    for split_name in splits:
        pairs = split_to_pairs.get(split_name, [])
        dst_dir = out_root / split_name / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            src_path: Path = pair["image"]
            dst_path = dst_dir / src_path.name
            if dst_path.exists():
                continue
            try:
                dst_path.symlink_to(src_path.resolve())
            except OSError as e:
                print(f"[WARN] link failed {src_path} -> {dst_path}: {e}")


# --- XML / 이미지 매칭 ---

import xml.etree.ElementTree as ET


def parse_voc_xml(xml_path: Path):
    """
    VOC 형식 XML 파싱 → bbox 리스트 반환
    반환: [{"class_name": "MA", "xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}, ...]
    """
    tree = ET.parse(xml_path)
    boxes = []
    seen = set()
    for obj in tree.findall("object"):
        raw_name = obj.find("name").text  # "ma"/"he"/"ex"/"se"
        if raw_name is None:
            continue
        raw_name = raw_name.strip()
        full_name = XML_CLASS_TO_NAME.get(raw_name)
        if full_name is None:
            # 모르는 클래스명은 스킵
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
        boxes.append({
            "class_name": full_name,  # "MA"/"HE"/"EX"/"SE"
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })
    return boxes


def find_seg_image_for_xml(xml_path: Path, seg_img_dirs=SEG_IMG_DIRS):
    """
    XML 파일명(stem)으로 segmentation image 폴더(train/valid/test/image)에서 실제 이미지 찾기.
    """
    stem = xml_path.stem
    for split_name, img_dir in seg_img_dirs.items():
        if not img_dir.exists():
            continue
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            img_path = img_dir / f"{stem}{ext}"
            if img_path.exists():
                return img_path, split_name
    return None, None


def load_pairs_for_ddr_split(ann_split: str):
    """
    lesion_detection/<ann_split> 안의 XML들에 대해,
    segmentation image에서 매칭되는 이미지 경로를 찾는다.
    """
    ann_dir = LESION_DET_ROOT / ann_split
    pairs = []
    for xml_path in sorted(ann_dir.glob("*.xml")):
        img_path, seg_split = find_seg_image_for_xml(xml_path)
        if img_path:
            pairs.append({
                "image": img_path,
                "xml": xml_path,
                "ann_split": ann_split,
                "seg_split": seg_split,
            })
    return pairs


def load_all_pairs():
    """train/valid/test 전체 pairs (benchmark 분할 무시하고 전부)."""
    all_pairs = []
    for ann_split in ("train", "valid", "test"):
        all_pairs.extend(load_pairs_for_ddr_split(ann_split))
    return all_pairs


def count_boxes_for_pair(pair) -> int:
    """해당 pair(XML)에 존재하는 bbox 개수."""
    xml_path = pair["xml"]
    boxes = parse_voc_xml(xml_path)
    return len(boxes)


# --- stratified split (DDR_re) ---

def _lesion_group_label(n_boxes: int) -> int:
    """
    bbox 개수로부터 간단한 그룹 레이블(0/1/2) 생성.
    0: 레이즌 없음, 1: 1~10개, 2: 11개 이상
    """
    if n_boxes == 0:
        return 0
    elif n_boxes <= 10:
        return 1
    else:
        return 2


def stratified_split_ddr_re(pairs, seed=DEFAULT_SEED, ratios=RE_SPLIT_RATIOS):
    """
    DDR 전체 pairs에 대해 train:test:val = 7:2:1 stratified split.
    FGART의 stratified_split_train_val_test 구조와 동일.
    """
    paths = list(pairs)
    n_boxes_list = [count_boxes_for_pair(p) for p in paths]
    labels = [_lesion_group_label(n) for n in n_boxes_list]

    # 1) train vs rest (val+test)
    total = len(paths)
    rest_size = 1.0 - ratios[0]  # 0.3
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=rest_size, random_state=seed
    )
    train_idx, rest_idx = next(sss1.split(paths, labels))
    train_pairs = [paths[i] for i in train_idx]
    rest_pairs = [paths[i] for i in rest_idx]
    rest_labels = [labels[i] for i in rest_idx]

    # 2) rest → test:val = 2:1 → 전체에서 test=0.2, val=0.1
    val_ratio = ratios[2] / (ratios[1] + ratios[2])  # 0.1 / 0.3 = 1/3
    test_ratio_in_rest = 1.0 - val_ratio  # 2/3

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio_in_rest, random_state=seed + 1
    )
    val_idx, test_idx = next(sss2.split(rest_pairs, rest_labels))
    val_pairs = [rest_pairs[i] for i in val_idx]
    test_pairs = [rest_pairs[i] for i in test_idx]

    print(
        f"[DDR_re split] total={total}, "
        f"train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}"
    )
    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }


# --- COCO 빌더 (4cls, 1cls 공통) ---

def build_coco_4cls(
    out_root: Path,
    split_to_pairs: Dict[str, List[dict]],
    splits,
):
    """
    DDR용 4cls COCO JSON 생성.
    - image_id, annotation id는 0-based
    - bbox: [x, y, w, h]
    - category name / id는 FGART와 동일 (MA/HE/EX/SE, 0~3)
    """
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
            img_path: Path = pair["image"]
            xml_path: Path = pair["xml"]

            w, h = _get_image_size(img_path)
            images.append({
                "id": image_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            })

            for box in parse_voc_xml(xml_path):
                cname = box["class_name"]   # "MA"/"HE"/"EX"/"SE"
                cid = CLASS_NAME_TO_ID.get(cname)
                if cid is None:
                    continue
                x1, y1 = box["xmin"], box["ymin"]
                x2, y2 = box["xmax"], box["ymax"]
                ww, hh = x2 - x1, y2 - y1
                if ww <= 0 or hh <= 0:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [float(x1), float(y1), float(ww), float(hh)],
                    "area": float(ww * hh),
                    "iscrowd": 0,
                })
                ann_id += 1

        (out_root / split_name).mkdir(parents=True, exist_ok=True)
        json_path = out_root / split_name / f"{split_name}.json"
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories_4cls,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

        print(
            f"[OK] 4cls COCO ({out_root.name}) {split_name}: "
            f"images={len(images)}, annotations={len(annotations)}"
        )


def build_coco_1cls_from_4cls(out_root_4cls: Path, out_root_1cls: Path, splits):
    """
    4cls JSON을 읽어 모든 category_id를 0으로 바꾸고 1cls 전용 루트에 저장.
    읽기: out_root_4cls/<split>/<split>.json
    쓰기: out_root_1cls/<split>/<split>.json
    """
    categories_1cls = [{"id": 0, "name": "lesion"}]

    for split_name in splits:
        json_path = out_root_4cls / split_name / f"{split_name}.json"
        if not json_path.exists():
            print(f"[WARN] 4cls not found: {json_path}, skip 1cls for {split_name}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
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


def print_split_summary(split_to_pairs: Dict[str, List[dict]], label: str, splits):
    """
    요약 통계: 각 split별 이미지 개수, 비율, 인스턴스(레이즌 bbox) 개수.
    FGART 준비 스크립트의 Summary 블록과 동일한 형태.
    """
    print(f"\n[Summary] {label} split별 이미지/레이즌 통계")
    total_imgs = sum(len(v) for v in split_to_pairs.values())
    for split_name in splits:
        pairs = split_to_pairs.get(split_name, [])
        n_imgs = len(pairs)
        ratio = (n_imgs / total_imgs * 100.0) if total_imgs > 0 else 0.0
        # 각 이미지별 VOC XML에서 bbox 개수 합산
        n_instances = 0
        for pair in pairs:
            xml_path = pair["xml"]
            n_instances += len(parse_voc_xml(xml_path))
        print(
            f"  {split_name}: images={n_imgs} ({ratio:.1f}%), "
            f"instances={n_instances}"
        )


# --- 메인 ---

def main(mode="both", seed=DEFAULT_SEED):
    """
    mode:
      - "bm":   DDR_bm만 처리 (기존 train/valid/test split)
      - "re":   DDR_re만 처리 (새 stratified split)
      - "both": 둘 다
    """
    print(f"[DDR] mode={mode}")

    # 1) DDR_bm
    if mode in ("bm", "both"):
        print("[DDR_bm] load_pairs_for_ddr_split()")
        split_to_pairs_bm = {
            s: load_pairs_for_ddr_split(s) for s in BM_SPLITS
        }
        for k, v in split_to_pairs_bm.items():
            print(f"  {k}: {len(v)}")

        print("[DDR_bm] create_split_dirs() — 4cls & 1cls (images/)")
        create_split_dirs(OUT_DDR_BM_4CLS, BM_SPLITS)
        create_split_dirs(OUT_DDR_BM_1CLS, BM_SPLITS)

        print("[DDR_bm] link_images_per_split() — 4cls & 1cls (symlinks)")
        link_images_per_split(OUT_DDR_BM_4CLS, split_to_pairs_bm, BM_SPLITS)
        link_images_per_split(OUT_DDR_BM_1CLS, split_to_pairs_bm, BM_SPLITS)

        print("[DDR_bm] build_coco_4cls()")
        build_coco_4cls(OUT_DDR_BM_4CLS, split_to_pairs_bm, BM_SPLITS)

        print("[DDR_bm] build_coco_1cls_from_4cls()")
        build_coco_1cls_from_4cls(OUT_DDR_BM_4CLS, OUT_DDR_BM_1CLS, BM_SPLITS)

        # 요약 통계
        print_split_summary(split_to_pairs_bm, label="DDR_bm", splits=BM_SPLITS)

    # 2) DDR_re
    if mode in ("re", "both"):
        print("[DDR_re] load_all_pairs()")
        all_pairs = load_all_pairs()
        print(f"  total pairs: {len(all_pairs)}")

        print("[DDR_re] stratified_split_ddr_re()")
        split_to_pairs_re = stratified_split_ddr_re(all_pairs, seed=seed, ratios=RE_SPLIT_RATIOS)

        for k, v in split_to_pairs_re.items():
            print(f"  {k}: {len(v)}")

        print("[DDR_re] create_split_dirs() — 4cls & 1cls (images/)")
        create_split_dirs(OUT_DDR_RE_4CLS, RE_SPLITS)
        create_split_dirs(OUT_DDR_RE_1CLS, RE_SPLITS)

        print("[DDR_re] link_images_per_split() — 4cls & 1cls (symlinks)")
        link_images_per_split(OUT_DDR_RE_4CLS, split_to_pairs_re, RE_SPLITS)
        link_images_per_split(OUT_DDR_RE_1CLS, split_to_pairs_re, RE_SPLITS)

        print("[DDR_re] build_coco_4cls()")
        build_coco_4cls(OUT_DDR_RE_4CLS, split_to_pairs_re, RE_SPLITS)

        print("[DDR_re] build_coco_1cls_from_4cls()")
        build_coco_1cls_from_4cls(OUT_DDR_RE_4CLS, OUT_DDR_RE_1CLS, RE_SPLITS)

        # 요약 통계
        print_split_summary(split_to_pairs_re, label="DDR_re", splits=RE_SPLITS)

    print("Done.")


if __name__ == "__main__":
    # 간단히: python ddr_prepare.py bm / re / both
    mode = "both"
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
    main(mode=mode)
