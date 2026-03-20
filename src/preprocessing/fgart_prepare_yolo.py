# src/preprocessing/fgart_prepare_yolo.py

"""
FGART(FGADR) COCO → YOLO 변환 스크립트.

입력:
  - 4cls COCO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_4scls/{train,val,test}/{split}.json
  - 1cls COCO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_1scls/{train,val,test}/{split}.json

출력:
  - 4cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_yolo_4cls/{train,val,test}/{images,labels}
  - 1cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_yolo_1cls/{train,val,test}/{images,labels}

이미지는 항상 심볼릭 링크를 사용하고, 복사 옵션은 제공하지 않는다.
YOLO 라벨 형식: class_id cx cy w_norm h_norm  (0-based class id, 0~1 정규화 좌표)
"""

from pathlib import Path
import json
import argparse

SPLITS = ("train", "val", "test")

# COCO 입력 루트
COCO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_4scls")
COCO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_1scls")

# YOLO 출력 루트
YOLO_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_yolo_4cls")
YOLO_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/FGART_yolo_1cls")


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO bbox [x, y, w, h] → YOLO [cx, cy, w_norm, h_norm]
    """
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return cx, cy, wn, hn


def convert_coco_to_yolo(coco_root: Path, yolo_root: Path, is_1cls: bool = False):
    """
    coco_root/<split>/<split>.json 을 읽어서
    yolo_root/<split>/{images,labels} 를 생성.

    - is_1cls=True 이면 category_id는 이미 0 하나라고 가정 (FGART_1scls)
    - 이미지는 항상 심볼릭 링크로 연결한다 (링크 실패 시에는 경고만 출력).
    """
    for split in SPLITS:
        coco_json = coco_root / split / f"{split}.json"
        if not coco_json.exists():
            print(f"[WARN] COCO JSON not found: {coco_json}")
            continue

        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images_by_id = {img["id"]: img for img in coco["images"]}

        anns_by_image = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            anns_by_image.setdefault(img_id, []).append(ann)

        # 입력 이미지 디렉터리 (fgart_prepare.py에서 만든 구조를 따른다)
        src_img_dir = coco_root / split / "images"

        # YOLO 출력 디렉터리
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
            except OSError as e:
                print(f"[WARN] symlink failed: {src_img_path} -> {dst_img_path} ({e})")

            anns = anns_by_image.get(img_id, [])
            label_lines = []
            for ann in anns:
                if is_1cls:
                    class_id = 0
                else:
                    class_id = int(ann["category_id"])  # 0~3 (MA/HE/EX/SE)

                cx, cy, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

            label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
            if label_lines:
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            else:
                # bbox 없는 이미지는 빈 파일 생성 (negative sample)
                label_path.write_text("", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["4cls", "1cls", "both"],
        default="both",
        help="어떤 버전을 YOLO로 변환할지 선택 (4cls / 1cls / both)",
    )
    args = parser.parse_args()

    if args.mode in ("4cls", "both"):
        print("[YOLO] 4cls 변환 시작")
        convert_coco_to_yolo(
            coco_root=COCO_4CLS_ROOT,
            yolo_root=YOLO_4CLS_ROOT,
            is_1cls=False,
        )

    if args.mode in ("1cls", "both"):
        print("[YOLO] 1cls 변환 시작")
        convert_coco_to_yolo(
            coco_root=COCO_1CLS_ROOT,
            yolo_root=YOLO_1CLS_ROOT,
            is_1cls=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()
