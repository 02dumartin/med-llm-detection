# src/visualization/ddr_overlay.py

"""
원본 DDR 데이터의 bbox 오버레이
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === 클래스 / 색 설정 (FGART와 동일) ===
CLASS_ORDER = ["ma", "he", "ex", "se"]
ID2NAME = {0: "MA", 1: "HE", 2: "EX", 3: "SE"}
NAME2ID = {"ma": 0, "he": 1, "ex": 2, "se": 3}
COLORS = {
    "ma": "#00FF00",  # MA
    "he": "#FF0000",  # HE
    "ex": "#FFFF00",  # EX
    "se": "#00FFFF",  # SE
}

# === DDR 경로 설정 ===
DDR_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR")

# detection 어노테이션(XML) 폴더
ANN_ROOT = DDR_ROOT / "lesion_detection"

# 이미지 폴더들 (노트북에서 쓰던 img_dirs와 비슷하게)
IMG_DIRS = {
    "seg_train": DDR_ROOT / "lesion_segmentation" / "train" / "image",
    "seg_valid": DDR_ROOT / "lesion_segmentation" / "valid" / "image",
    "seg_test": DDR_ROOT / "lesion_segmentation" / "test" / "image",
}

# 출력 위치 (FGADR처럼 /data 쪽에 두는 걸 추천)
OUT_DIR = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR/ddr_gt_overlay")


# === 유틸 함수들 ===

def find_image(stem: str, split_hint: str | None = None):
    """
    DDR 노트북의 find_image와 비슷한 역할:
    - 모든 IMG_DIRS에서 stem이 같은 이미지(.jpg/.jpeg/.png)를 찾음.
    """
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    # split_hint가 있으면 해당 split만 우선 검색
    if split_hint:
        key = f"seg_{split_hint}"
        img_dir = IMG_DIRS.get(key)
        if img_dir and img_dir.exists():
            for ext in exts:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    return p
    # fallback: 전체 검색
    for img_dir in IMG_DIRS.values():
        if not img_dir.exists():
            continue
        for ext in exts:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def parse_voc_xml(xml_path: Path):
    """
    VOC 형식 XML을 파싱해서 bbox 리스트를 반환.
    각 bbox는 dict로 {class, xmin, ymin, xmax, ymax}
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text  # ex, he, ma, se (소문자일 가능성 높음)
        if name is None:
            continue
        cls = name.lower()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append(
            {
                "class": cls,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )
    return boxes


def extract_bboxes_ddr(xml_path: Path):
    """
    DDR detection XML에서 bbox 리스트 (MA/HE/EX/SE만) 추출.
    """
    boxes = parse_voc_xml(xml_path)
    filtered = []
    seen = set()
    for box in boxes:
        cls = box["class"].lower()
        if cls not in COLORS:             # ma/he/ex/se 이외는 스킵
            continue
        key = (cls, box["xmin"], box["ymin"], box["xmax"], box["ymax"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(box)
    return filtered


# === 시각화 함수 ===

def draw_ddr_overlay(ax, xml_path: Path, legend: bool = True):
    """
    DDR detection 이미지 1장에 대해 MA/HE/EX/SE bbox를 ax에 오버레이.
    - legend=False면 범례를 추가하지 않음.
    """
    stem = xml_path.stem
    split_hint = xml_path.parent.name  # train/valid/test
    img_path = find_image(stem, split_hint=split_hint)
    if img_path is None:
        print(f"[WARN] image not found for {xml_path.name}")
        return 0

    boxes = extract_bboxes_ddr(xml_path)
    if not boxes:
        print(f"[INFO] no bboxes (MA/HE/EX/SE) for {xml_path.name}")
    img = Image.open(img_path)

    ax.imshow(img)

    used = 0
    for box in boxes:
        cls = box["class"].lower()
        if cls not in COLORS:
            continue
        color = COLORS[cls]
        x = box["xmin"]
        y = box["ymin"]
        w = box["xmax"] - x
        h = box["ymax"] - y
        rect = patches.Rectangle(
            (x, y), w, h,
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


def visualize_with_bbox_ddr(xml_path: Path, figsize=(12, 12)):
    """
    DDR detection 이미지 1장에 대해 MA/HE/EX/SE bbox를 오버레이해서 PNG로 저장.
    - xml_path: lesion_detection/train|valid|test 안의 XML 경로
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    draw_ddr_overlay(ax, xml_path, legend=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{xml_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


# === 전체 실행 ===

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_paths = []
    for split in ["train", "valid", "test"]:
        d = ANN_ROOT / split
        if d.exists():
            xml_paths.extend(sorted(d.glob("*.xml")))
    print(f"Total DDR XMLs: {len(xml_paths)}")

    for i, xml_path in enumerate(xml_paths):
        visualize_with_bbox_ddr(xml_path)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(xml_paths)} done")


if __name__ == "__main__":
    main()
