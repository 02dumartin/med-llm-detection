"""
FGART GT bbox overlay 재생성 스크립트.
- min_box_size=16 적용된 bbox 기준
- 각 이미지에 MA/HE/EX/SE 색상 구분 레전드 포함
- 출력: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART/fgart_gt_overlay_re/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.fgart_prepare import extract_bboxes, IMG_ROOT, CLASS_FULL_TO_ID

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART/fgart_gt_overlay_re")

COLORS = {
    "Microaneurysms": "#00FF00",
    "Hemorrhage":     "#FF0000",
    "HardExudate":    "#FFFF00",
    "SoftExudate":    "#00FFFF",
}
LEGEND_LABELS = {
    "Microaneurysms": "MA",
    "Hemorrhage":     "HE",
    "HardExudate":    "EX",
    "SoftExudate":    "SE",
}


def draw_overlay(img_name: str, min_area: int = 5, min_box_size: int = 16):
    img_path = IMG_ROOT / img_name
    if not img_path.exists():
        print(f"[WARN] not found: {img_path}")
        return

    bboxes = extract_bboxes(img_name, min_area=min_area, min_box_size=min_box_size)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    with Image.open(img_path) as im:
        ax.imshow(im)

    seen_cls = set()
    for cls_full, x1, y1, x2, y2 in bboxes:
        color = COLORS.get(cls_full)
        if color is None:
            continue
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        seen_cls.add(cls_full)

    # 레전드 (항상 4개 전부 표시, Line2D 스타일)
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS[c], linewidth=4, label=LEGEND_LABELS[c])
        for c in ["Microaneurysms", "Hemorrhage", "HardExudate", "SoftExudate"]
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right",
                  fontsize=10, framealpha=0.7)

    ax.axis("off")
    ax.set_title(img_name, fontsize=8, pad=2)

    out_path = OUT_DIR / img_name
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_paths = sorted([p for p in IMG_ROOT.iterdir() if p.suffix.lower() == ".png"])
    total = len(img_paths)
    print(f"총 {total}장 overlay 생성 시작 → {OUT_DIR}")

    for i, img_path in enumerate(img_paths):
        draw_overlay(img_path.name)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total} 완료")

    print("overlay 생성 완료")


if __name__ == "__main__":
    main()
