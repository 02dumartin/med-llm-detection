# src/preprocessing/ddr_prepare_yolo.py

"""
DDR COCO → YOLO 변환 스크립트.

입력 (COCO는 모두 /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data 하위, JSON 전용):
  - DDR_bm 4cls COCO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_4cls/{train,valid,test}/{split}.json
  - DDR_bm 1cls COCO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_1cls/{train,valid,test}/{split}.json
  - DDR_re 4cls COCO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_4cls/{train,val,test}/{split}.json
  - DDR_re 1cls COCO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_1cls/{train,val,test}/{split}.json

입력 이미지 심볼릭 링크 (ddr_prepare.py에서 생성, FGART_4scls와 동일 구조로 /data 하위):
  - DDR_bm images: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_4cls/{train,valid,test}/images
  - DDR_re images: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_4cls/{train,val,test}/images

출력 (YOLO도 /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data 하위):
  - DDR_bm 4cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_yolo_4cls/{train,valid,test}/{images,labels}
  - DDR_bm 1cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_yolo_1cls/{train,valid,test}/{images,labels}
  - DDR_re 4cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_yolo_4cls/{train,val,test}/{images,labels}
  - DDR_re 1cls YOLO: /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_yolo_1cls/{train,val,test}/{images,labels}

YOLO 라벨 형식:
  class_id cx cy w_norm h_norm  (0-based class id, 0~1 정규화 좌표)
"""

from pathlib import Path
import json
import argparse

# DDR_bm splits (valid 유지)
BM_SPLITS = ("train", "valid", "test")
# DDR_re splits (FGART 스타일)
RE_SPLITS = ("train", "val", "test")

# COCO 입력 루트 (데이터 루트 하위, JSON 전용)
COCO_BM_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_4cls")
COCO_BM_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_1cls")

COCO_RE_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_4cls")
COCO_RE_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_1cls")

# 이미지 심볼릭 링크 루트 (FGART_4scls와 동일하게 /data 하위 COCO 루트를 그대로 사용)
IMG_BM_ROOT = COCO_BM_4CLS_ROOT
IMG_RE_ROOT = COCO_RE_4CLS_ROOT

# YOLO 출력 루트 (데이터 루트 하위)
YOLO_BM_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_yolo_4cls")
YOLO_BM_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_bm_yolo_1cls")

YOLO_RE_4CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_yolo_4cls")
YOLO_RE_1CLS_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_re_yolo_1cls")


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


def convert_coco_to_yolo(
    coco_root: Path,
    img_root: Path,
    yolo_root: Path,
    splits,
    is_1cls: bool = False,
):
    """
    coco_root/<split>/<split>.json 을 읽어서
    yolo_root/<split>/{images,labels} 를 생성.

    - is_1cls=True 이면 category_id는 이미 0 하나라고 가정
    - 이미지는 항상 심볼릭 링크로 연결한다 (링크 실패 시에는 경고만 출력).
    """
    for split in splits:
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

        # 입력 이미지 디렉터리 (ddr_prepare.py에서 /data/.../DDR_*_4cls/<split>/images 로 만든 구조를 따른다)
        src_img_dir = img_root / split / "images"
        if not src_img_dir.exists():
            # DDR에서는 COCO JSON 만들 때 심볼릭 링크를 따로 안 만들고,
            # segmentation image 원본에서 직접 읽었으므로
            # 여기서는 images 디렉토리 대신 segmentation root에서 링크를 만들어도 됨.
            # 간단하게는 segmentation image를 직접 참조해서 링크 생성.
            print(f"[WARN] src_img_dir not found: {src_img_dir} (skip symlink)")
            src_img_dir = None

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

            if src_img_dir is not None:
                src_img_path = src_img_dir / file_name
            else:
                # src_img_dir가 없으면 그냥 label만 생성 (이미지는 별도 관리)
                src_img_path = None

            dst_img_path = img_out_dir / file_name

            if src_img_path is not None and src_img_path.exists():
                if not dst_img_path.exists():
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
        "--dataset",
        choices=["bm", "re", "both"],
        default="both",
        help="어떤 DDR 세트를 YOLO로 변환할지 선택 (bm / re / both)",
    )
    parser.add_argument(
        "--mode",
        choices=["4cls", "1cls", "both"],
        default="both",
        help="어떤 버전을 YOLO로 변환할지 선택 (4cls / 1cls / both)",
    )
    args = parser.parse_args()

    if args.dataset in ("bm", "both"):
        if args.mode in ("4cls", "both"):
            print("[YOLO] DDR_bm 4cls 변환 시작")
            convert_coco_to_yolo(
                coco_root=COCO_BM_4CLS_ROOT,
                img_root=IMG_BM_ROOT,
                yolo_root=YOLO_BM_4CLS_ROOT,
                splits=BM_SPLITS,
                is_1cls=False,
            )
        if args.mode in ("1cls", "both"):
            print("[YOLO] DDR_bm 1cls 변환 시작")
            convert_coco_to_yolo(
                coco_root=COCO_BM_1CLS_ROOT,
                img_root=IMG_BM_ROOT,
                yolo_root=YOLO_BM_1CLS_ROOT,
                splits=BM_SPLITS,
                is_1cls=True,
            )

    if args.dataset in ("re", "both"):
        if args.mode in ("4cls", "both"):
            print("[YOLO] DDR_re 4cls 변환 시작")
            convert_coco_to_yolo(
                coco_root=COCO_RE_4CLS_ROOT,
                img_root=IMG_RE_ROOT,
                yolo_root=YOLO_RE_4CLS_ROOT,
                splits=RE_SPLITS,
                is_1cls=False,
            )
        if args.mode in ("1cls", "both"):
            print("[YOLO] DDR_re 1cls 변환 시작")
            convert_coco_to_yolo(
                coco_root=COCO_RE_1CLS_ROOT,
                img_root=IMG_RE_ROOT,
                yolo_root=YOLO_RE_1CLS_ROOT,
                splits=RE_SPLITS,
                is_1cls=True,
            )

    print("Done.")


if __name__ == "__main__":
    main()