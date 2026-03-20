# src/visualization/fgart_overlay.py

"""
FGART GT bbox 오버레이.
기본: 변환된 4cls COCO JSON 기반으로 bbox를 그린다 (fgart_prepare.py에서 생성한 결과와 동일).
"""

from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# === 클래스 / 색 설정 ===
CLASS_ORDER = ["ma", "he", "ex", "se"]
ID2NAME = {0: "MA", 1: "HE", 2: "EX", 3: "SE"}
NAME2ID = {"ma": 0, "he": 1, "ex": 2, "se": 3}
COLORS = {
    "ma": "#00FF00",  # MA
    "he": "#FF0000",  # HE
    "ex": "#FFFF00",  # EX
    "se": "#00FFFF",  # SE
}

# FGART 경로
root = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART/Seg-set")
IMG_DIR = root / "Original_Images"
OUT_DIR = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART/fgart_gt_overlay")

# 4cls COCO JSON 기반 bbox 사용
COCO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_4scls")
CID_TO_CLS = {0: "ma", 1: "he", 2: "ex", 3: "se"}


MASK_DIRS = {
    "Microaneurysms": root / "Microaneurysms_Masks",
    "Hemorrhage": root / "Hemohedge_Masks",
    "HardExudate": root / "HardExudate_Masks",
    "SoftExudate": root / "SoftExudate_Masks",
    # IRMA, NV 제외
}

try:
    from skimage.measure import label, regionprops
    def get_regions(mask):
        lbl = label(mask, connectivity=2)
        return regionprops(lbl)
except Exception:
    from scipy.ndimage import label as ndi_label, find_objects
    def get_regions(mask):
        lbl, _ = ndi_label(mask)
        slices = find_objects(lbl)
        regions = []
        for sl in slices:
            if sl is None:
                continue
            rmin, rmax = sl[0].start, sl[0].stop
            cmin, cmax = sl[1].start, sl[1].stop
            area = (rmax - rmin) * (cmax - cmin)
            regions.append(type("R", (), {"bbox": (rmin, cmin, rmax, cmax), "area": area}))
        return regions

def get_bboxes_from_coco_4cls(file_name: str) -> list[tuple[str, int, int, int, int]]:
    """
    4cls COCO JSON에서 해당 이미지의 bbox를 로드.
    Returns: [(cls_abbr, x1, y1, x2, y2), ...] (cls_abbr: ma/he/ex/se)
    """
    bboxes = []
    for split in ("train", "val", "test"):
        p = COCO_4CLS_ROOT / split / f"{split}.json"
        if not p.exists():
            continue
        with open(p, "r") as f:
            coco = json.load(f)
        img_list = {im["file_name"]: im for im in coco.get("images", [])}
        if file_name not in img_list:
            continue
        im_info = img_list[file_name]
        im_id = im_info["id"]
        for ann in coco.get("annotations", []):
            if ann["image_id"] != im_id:
                continue
            cid = ann["category_id"]
            cls = CID_TO_CLS.get(cid, "ma")
            x, y, w, h = ann["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            bboxes.append((cls, x1, y1, x2, y2))
        break
    return bboxes


def get_all_image_names_from_coco_4cls() -> list[str]:
    """4cls COCO JSON에 등장하는 모든 이미지 파일명 리스트 (중복 제거)."""
    seen = set()
    names = []
    for split in ("train", "val", "test"):
        p = COCO_4CLS_ROOT / split / f"{split}.json"
        if not p.exists():
            continue
        with open(p, "r") as f:
            coco = json.load(f)
        for im in coco.get("images", []):
            fn = im["file_name"]
            if fn not in seen:
                seen.add(fn)
                names.append(fn)
    return sorted(names)


def get_images_per_split_from_coco_4cls() -> list[tuple[str, str]]:
    """4cls COCO JSON에서 (split, file_name) 리스트 반환. split 순서: train, val, test."""
    result = []
    for split in ("train", "val", "test"):
        p = COCO_4CLS_ROOT / split / f"{split}.json"
        if not p.exists():
            continue
        with open(p, "r") as f:
            coco = json.load(f)
        for im in coco.get("images", []):
            result.append((split, im["file_name"]))
    return result


def load_binary_mask(mpath):
    """마스크 이미지를 항상 2D 바이너리 배열로 로드."""
    from PIL import Image
    import numpy as np

    with Image.open(mpath) as im:
        arr = np.array(im)
    if arr.ndim == 3:        # (H, W, C)인 경우
        arr = arr[..., 0]    # 첫 채널만 사용 (또는 arr.mean(axis=-1))
    mask = (arr > 0).astype(np.uint8)
    return mask

def extract_bboxes(img_name, min_area=5):
    bboxes = []
    for cls, mdir in MASK_DIRS.items():
        mpath = mdir / img_name
        if not mpath.exists():
            continue
        mask = load_binary_mask(mpath)   # << 여기
        if mask.max() == 0:
            continue
        regions = get_regions(mask)
        for r in regions:
            if getattr(r, "area", 0) < min_area:
                continue
            rmin, cmin, rmax, cmax = r.bbox
            bboxes.append((cls, cmin, rmin, cmax, rmax))
    return bboxes

def draw_fgart_overlay(
    ax,
    img_name: str,
    min_area: int = 5,
    legend: bool = True,
    use_coco: bool = True,
):
    """
    FGART Seg-set 이미지 1장에 대해 MA/HE/EX/SE bbox를 ax에 오버레이.
    - use_coco=True: 4cls COCO JSON 기반 bbox 사용 (기본)
    - use_coco=False: 마스크에서 bbox 추출
    - legend=False면 범례를 추가하지 않음.
    """
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        print(f"[WARN] image not found: {img_path}")
        return 0

    if use_coco:
        bboxes = get_bboxes_from_coco_4cls(img_name)  # [(cls_abbr, x1,y1,x2,y2), ...]
    else:
        bboxes = extract_bboxes(img_name, min_area=min_area)  # [(cls_full, x1,y1,x2,y2), ...]

    if not bboxes:
        print(f"[INFO] no bboxes for {img_name}")
    img = Image.open(img_path)

    ax.imshow(img)

    used = 0
    for item in bboxes:
        cls_or_full, x1, y1, x2, y2 = item
        if use_coco:
            cls = cls_or_full  # 이미 ma/he/ex/se
        else:
            cls_full = cls_or_full
            if cls_full.lower().startswith("micro"):
                cls = "ma"
            elif cls_full.lower().startswith("hemo"):
                cls = "he"
            elif cls_full.lower().startswith("hard"):
                cls = "ex"
            elif cls_full.lower().startswith("soft"):
                cls = "se"
            else:
                continue

        if cls not in COLORS:
            continue

        color = COLORS[cls]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        used += 1

    if legend:
        legend_elements = [
            plt.Line2D([0], [0], color=COLORS[c], linewidth=4, label=c.upper())
            for c in CLASS_ORDER
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    ax.axis("off")
    return used


def visualize_with_bbox_fgart(
    img_name: str,
    min_area: int = 5,
    figsize=(12, 12),
    use_coco: bool = True,
    subdir: str | None = None,
):
    """
    FGART 이미지 1장에 대해 MA/HE/EX/SE bbox를 오버레이해서 PNG로 저장.
    use_coco=True: 4cls COCO JSON 기반 (기본)
    subdir: None이면 OUT_DIR/img_name, 지정 시 OUT_DIR/subdir/img_name (예: "train", "val", "test")
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    draw_fgart_overlay(ax, img_name, min_area=min_area, legend=True, use_coco=use_coco)
    out_folder = OUT_DIR / subdir if subdir else OUT_DIR
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / img_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OK] saved {out_path}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = get_images_per_split_from_coco_4cls()
    print(f"Total FGART images (4cls): {len(items)}")

    for i, (split, img_name) in enumerate(items):
        visualize_with_bbox_fgart(img_name, min_area=5, use_coco=True, subdir=split)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(items)} done")

if __name__ == "__main__":
    main()
