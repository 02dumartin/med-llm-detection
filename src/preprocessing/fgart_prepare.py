# src/preprocessing/fgart_prepare.py

"""
FGART(FGADR) 데이터 준비:스플릿 → 심볼릭 링크 → bbox 추출 → 4cls COCO JSON → 1cls COCO JSON.
bbox 추출은 fgart_overlay의 extract_bboxes + 마스크 로드 로직 재사용.
image_id, annotation id는 0-based.
"""

from pathlib import Path
import json
import sys

from sklearn.model_selection import StratifiedShuffleSplit

# 프로젝트 루트에서 visualization 모듈 import
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.fgart_overlay import load_binary_mask, get_regions


# --- 경로 / 상수 ---
FGART_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART")
IMG_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART/Seg-set/Original_Images")
MASK_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART/Seg-set")
OUT_ROOT_4CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_4scls")   # 4cls: train/val/test
OUT_ROOT_1CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_1scls")   # 1cls: train/val/test

MASK_DIRS = {
    "Microaneurysms": MASK_ROOT / "Microaneurysms_Masks",
    "Hemorrhage": MASK_ROOT / "Hemohedge_Masks",
    "HardExudate": MASK_ROOT / "HardExudate_Masks",
    "SoftExudate": MASK_ROOT / "SoftExudate_Masks",
}

# 4cls: MA=0, HE=1, EX=2, SE=3
CLASS_FULL_TO_ID = {
    "Microaneurysms": 0,
    "Hemorrhage": 1,
    "HardExudate": 2,
    "SoftExudate": 3,
}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.7, 0.1, 0.2)
DEFAULT_SEED = 42
MIN_BOX_SIZE = 16  # 소형 병변(MA 등) bbox 최소 크기(px). 이보다 작으면 중심 고정 후 확장.


def get_image_list(extensions=(".png", ".jpg", ".jpeg")):
    """Original_Images 아래 이미지 경로 목록 반환 (정렬)."""
    out = []
    for p in IMG_ROOT.iterdir():
        if p.suffix.lower() in extensions:
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def create_split_dirs(out_root, splits=SPLITS):
    """out_root/train/images, out_root/val/images, out_root/test/images 생성"""
    for s in splits:
        (out_root / s / "images").mkdir(parents=True, exist_ok=True)


def link_images_per_split(out_root, split_to_images):
    """각 split별 이미지 리스트를 out_root/<split>/images/ 에 심볼릭 링크로 넣음"""
    for split_name, paths in split_to_images.items():
        dst_dir = out_root / split_name / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_path in paths:
            dst_path = dst_dir / src_path.name
            if dst_path.is_symlink() or dst_path.exists():
                dst_path.unlink()
            try:
                dst_path.symlink_to(src_path.resolve())
            except OSError as e:
                print(f"[WARN] link failed {src_path} -> {dst_path}: {e}")


def _apply_min_box_size(x1, y1, x2, y2, img_w, img_h, min_size=MIN_BOX_SIZE):
    """bbox 너비/높이가 min_size 미만이면 중심 고정 후 최소 크기로 확장. 이미지 경계 클리핑."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(x2 - x1, min_size)
    h = max(y2 - y1, min_size)
    x1_new = max(0, int(cx - w / 2))
    y1_new = max(0, int(cy - h / 2))
    x2_new = min(img_w, int(cx + w / 2))
    y2_new = min(img_h, int(cy + h / 2))
    return x1_new, y1_new, x2_new, y2_new


def extract_bboxes(img_name, min_area=5, min_box_size=MIN_BOX_SIZE):
    """마스크에서 bbox 추출. 소형 bbox는 min_box_size(px) 이상으로 패딩.
    반환: (cls_full, x1, y1, x2, y2) 리스트."""
    from PIL import Image as _PIL_Image
    # 이미지 크기 파악 (클리핑 기준)
    img_path = IMG_ROOT / img_name
    if img_path.exists():
        with _PIL_Image.open(img_path) as _im:
            img_w, img_h = _im.size
    else:
        img_w, img_h = 1280, 1280  # fallback

    bboxes = []
    for cls, mdir in MASK_DIRS.items():
        mpath = mdir / img_name
        if not mpath.exists():
            continue
        mask = load_binary_mask(mpath)
        if mask.max() == 0:
            continue
        regions = get_regions(mask)
        for r in regions:
            if getattr(r, "area", 0) < min_area:
                continue
            rmin, cmin, rmax, cmax = r.bbox
            x1, y1, x2, y2 = cmin, rmin, cmax, rmax
            if min_box_size > 0:
                x1, y1, x2, y2 = _apply_min_box_size(x1, y1, x2, y2, img_w, img_h, min_box_size)
            bboxes.append((cls, x1, y1, x2, y2))
    return bboxes


def extract_all_bboxes(split_to_images, min_area=5):
    """split별 이미지 path 리스트에 대해 bbox 수집. 반환: {split: {img_name: [(cls, x1,y1,x2,y2), ...]}}"""
    out = {}
    for split_name, paths in split_to_images.items():
        out[split_name] = {}
        for src_path in paths:
            name = src_path.name
            out[split_name][name] = extract_bboxes(name, min_area=min_area)
    return out


def _class_pattern_label(bboxes: list) -> str:
    """이미지의 bbox 리스트로부터 클래스 존재 패턴 라벨 생성.

    각 클래스(MA/HE/EX/SE)의 존재 여부를 비트 플래그로 인코딩하고,
    SE 희소 클래스는 'many'(5개 초과) / 'few'(1~5) / 'none'(0)으로 세분화한다.

    예: MA 있고 HE 있고 EX 없고 SE 없으면 → "1100_none"
    """
    if not bboxes:
        return "0000_none"

    cls_counts = {0: 0, 1: 0, 2: 0, 3: 0}   # MA HE EX SE
    for cls_full, *_ in bboxes:
        cid = CLASS_FULL_TO_ID.get(cls_full)
        if cid is not None:
            cls_counts[cid] += 1

    # 각 클래스 존재 여부 (0/1)
    flags = "".join(str(int(cls_counts[i] > 0)) for i in range(4))

    # SE(희소 클래스)를 추가 세분화: none / few(1~5) / many(6+)
    se_n = cls_counts[3]
    if se_n == 0:
        se_bucket = "none"
    elif se_n <= 5:
        se_bucket = "few"
    else:
        se_bucket = "many"

    return f"{flags}_{se_bucket}"


def stratified_split_train_val_test(image_list, min_area=5, seed=DEFAULT_SEED, ratios=SPLIT_RATIOS):
    """클래스 존재 패턴 기반 multi-class stratified split.

    절차:
    1) 모든 이미지에 대해 extract_bboxes를 돌려 클래스별 존재 패턴을 문자열 라벨로 인코딩
       - 각 클래스(MA/HE/EX/SE) 존재 여부 + SE 희소 클래스 세분화
       - 희소 패턴(<5장)은 가장 가까운 패턴으로 병합하여 StratifiedShuffleSplit 가능하게 처리
    2) sklearn StratifiedShuffleSplit으로
       - train vs (val+test)
       - (val+test)에서 val vs test
    """
    paths = list(image_list)

    # 1) 클래스 패턴 라벨 계산
    print("[stratified] analyzing per-class lesion patterns per image...")
    raw_labels = []
    for p in paths:
        bboxes = extract_bboxes(p.name, min_area=min_area)
        raw_labels.append(_class_pattern_label(bboxes))

    # 희소 패턴 병합: 5장 미만인 패턴은 SE 세분화 제거 후 flags만 사용
    from collections import Counter
    pattern_counts = Counter(raw_labels)
    labels = []
    for lbl in raw_labels:
        if pattern_counts[lbl] < 5:
            # SE 세분화 제거 → flags 4자리만 사용
            labels.append(lbl.split("_")[0])
        else:
            labels.append(lbl)

    # 재병합 후에도 희소한 패턴은 'has_lesion' / 'no_lesion' 으로 최종 폴백
    pattern_counts2 = Counter(labels)
    final_labels = []
    for lbl in labels:
        if pattern_counts2[lbl] < 3:
            final_labels.append("has_lesion" if lbl != "0000" else "no_lesion")
        else:
            final_labels.append(lbl)

    unique = sorted(set(final_labels))
    print(f"[stratified] 최종 패턴 종류: {len(unique)}개")
    for u in unique:
        print(f"  {u}: {final_labels.count(u)}장")

    def _merge_rare(label_list, min_count=2):
        """min_count 미만 패턴을 점진적으로 병합."""
        from collections import Counter
        lbs = list(label_list)
        for _ in range(5):
            cnt = Counter(lbs)
            updated = []
            for lb in lbs:
                if cnt[lb] < min_count:
                    # SE 세분화 제거
                    base = lb.split("_")[0]
                    if base != lb and cnt.get(base, 0) >= min_count:
                        updated.append(base)
                    else:
                        updated.append("has_lesion" if lb != "0000_none" else "no_lesion")
                else:
                    updated.append(lb)
            if Counter(updated) == Counter(lbs):
                break
            lbs = updated
        return lbs

    # 2) train vs rest (val+test)
    total = len(paths)
    test_size_rest = 1.0 - ratios[0]
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size_rest, random_state=seed
    )
    train_idx, rest_idx = next(sss1.split(paths, final_labels))
    train_paths = [paths[i] for i in train_idx]
    rest_paths  = [paths[i] for i in rest_idx]
    rest_labels_raw = [final_labels[i] for i in rest_idx]

    # rest 내에서도 희소 패턴 재병합 (2단계 분할을 위해)
    rest_labels = _merge_rare(rest_labels_raw, min_count=2)

    # 3) rest → val / test
    val_in_rest = ratios[1] / (ratios[1] + ratios[2])
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - val_in_rest, random_state=seed + 1
    )
    val_idx, test_idx = next(sss2.split(rest_paths, rest_labels))
    val_paths  = [rest_paths[i] for i in val_idx]
    test_paths = [rest_paths[i] for i in test_idx]

    print(
        f"[stratified] total={total}, "
        f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}"
    )
    return {
        "train": train_paths,
        "val":   val_paths,
        "test":  test_paths,
    }


def _get_image_size(img_path):
    """이미지 (width, height). PIL 사용"""
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size  # (width, height)


def build_coco_4cls(out_root, split_to_images, split_to_bboxes, categories_4cls=None):
    """4cls COCO JSON 생성. image_id, annotation id는 0-based. bbox는 [x, y, width, height]"""
    if categories_4cls is None:
        categories_4cls = [
            {"id": 0, "name": "MA"},
            {"id": 1, "name": "HE"},
            {"id": 2, "name": "EX"},
            {"id": 3, "name": "SE"},
        ]
    for split_name in SPLITS:
        paths = split_to_images.get(split_name, [])
        bbox_map = split_to_bboxes.get(split_name, {})
        images = []
        annotations = []
        ann_id = 0
        for image_id, src_path in enumerate(paths):
            name = src_path.name
            # 링크된 경로에서 크기 읽기 (원본과 동일)
            link_path = out_root / split_name / "images" / name
            if link_path.exists():
                w, h = _get_image_size(link_path)
            else:
                w, h = _get_image_size(src_path)
            images.append({
                "id": image_id,
                "file_name": name,
                "width": w,
                "height": h,
            })
            for cls_full, x1, y1, x2, y2 in bbox_map.get(name, []):
                cid = CLASS_FULL_TO_ID.get(cls_full)
                if cid is None:
                    continue
                x, y = float(x1), float(y1)
                ww, hh = float(x2 - x1), float(y2 - y1)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [x, y, ww, hh],
                    "area": ww * hh,
                    "iscrowd": 0,
                })
                ann_id += 1
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories_4cls,
        }
        json_path = out_root / split_name / f"{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(f"[OK] 4cls COCO: {json_path} (images={len(images)}, annotations={len(annotations)})")


def build_coco_1cls_from_4cls(out_root_4cls, out_root_1cls, splits=SPLITS):
    """4cls JSON을 읽어 모든 category_id를 0으로 바꾸고 1cls 전용 루트에 저장.
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
        for ann in coco["annotations"]:
            ann["category_id"] = 0
        coco["categories"] = categories_1cls
        (out_root_1cls / split_name).mkdir(parents=True, exist_ok=True)
        out_path = out_root_1cls / split_name / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(f"[OK] 1cls COCO: {out_path} (images={len(coco['images'])}, annotations={len(coco['annotations'])})")

def main(seed=DEFAULT_SEED, min_area=5):
    print("[1] get_image_list()")
    image_list = get_image_list()
    print(f"    total images: {len(image_list)}")

    print("[2] stratified_split_train_val_test()")
    split_to_images = stratified_split_train_val_test(
        image_list, min_area=min_area, seed=seed, ratios=SPLIT_RATIOS
    )
    for k, v in split_to_images.items():
        print(f"    {k}: {len(v)}")

    print("[3] create_split_dirs() — 4cls & 1cls")
    create_split_dirs(OUT_ROOT_4CLS)
    create_split_dirs(OUT_ROOT_1CLS)

    print("[4] link_images_per_split() — 4cls & 1cls")
    link_images_per_split(OUT_ROOT_4CLS, split_to_images)
    link_images_per_split(OUT_ROOT_1CLS, split_to_images)

    print("[5] extract_all_bboxes()")
    split_to_bboxes = extract_all_bboxes(split_to_images, min_area=min_area)

    print("[6] build_coco_4cls() → FGART_4scls")
    build_coco_4cls(OUT_ROOT_4CLS, split_to_images, split_to_bboxes)

    print("[7] build_coco_1cls_from_4cls() → FGART_1scls")
    build_coco_1cls_from_4cls(OUT_ROOT_4CLS, OUT_ROOT_1CLS)

    # 요약 통계: 각 폴더별 이미지 개수, 비율, 인스턴스(레이즌 bbox) 개수
    print("\n[Summary] split별 이미지/레이즌 통계")
    total_imgs = sum(len(v) for v in split_to_images.values())
    for split_name in SPLITS:
        imgs = split_to_images.get(split_name, [])
        bmap = split_to_bboxes.get(split_name, {})
        n_imgs = len(imgs)
        ratio = (n_imgs / total_imgs * 100.0) if total_imgs > 0 else 0.0
        n_instances = sum(len(bmap.get(p.name, [])) for p in imgs)
        print(
            f"  {split_name}: images={n_imgs} ({ratio:.1f}%), "
            f"instances={n_instances}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
