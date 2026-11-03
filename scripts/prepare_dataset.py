"""Convert HIT-UAV (COCO-style JSON) into a person-only YOLO dataset.

Output layout:
    data/processed/hit-uav-person/
        images/{train,holdout,val,test}/*.jpg
        labels/{train,holdout,val,test}/*.txt   (one line per person: "0 xc yc w h", normalized)
        data.yaml

Splits:
    train   - official train minus holdout, used to fit the detector
    holdout - part of official train, used to pick the best epoch / settings
    val     - official val, kept untouched for the reliability layer (QARL)
    test    - official test, used once for the final evaluation
"""
import argparse
import json
import random
import shutil
from pathlib import Path

PERSON_ID = 0            # category id of "Person" in the HIT-UAV JSON files
SPLITS = ["train", "val", "test"]


def coco_box_to_yolo(box, img_w, img_h):
    """COCO box [x_min, y_min, w, h] in pixels -> YOLO box [xc, yc, w, h] in 0-1."""
    x_min, y_min, w, h = box
    xc = (x_min + w / 2) / img_w
    yc = (y_min + h / 2) / img_h
    return xc, yc, w / img_w, h / img_h


def sequence_id(filename: str) -> str:
    """'0_100_30_0_03280.jpg' -> '0_100_30_0' (day/night flag, altitude, camera angle, ...).

    Images with the same prefix come from the same flight and are often consecutive
    frames that look almost identical, so they must stay on the same side of a split.
    """
    return "_".join(filename.split("_")[:4])


def pick_holdout(images, fraction: float, seed: int):
    """Pick whole sequences at random until they hold at least `fraction` of the images."""
    sequences = sorted({sequence_id(img["filename"]) for img in images})
    random.Random(seed).shuffle(sequences)

    target = fraction * len(images)
    chosen, count = set(), 0
    for seq in sequences:
        if count >= target:
            break
        chosen.add(seq)
        count += sum(sequence_id(img["filename"]) == seq for img in images)
    return {img["id"] for img in images if sequence_id(img["filename"]) in chosen}


def convert_split(src_root: Path, out_root: Path, split: str, holdout_fraction: float, seed: int):
    with open(src_root / "normal_json" / "annotations" / f"{split}.json") as f:
        coco = json.load(f)

    # Only the official train split gives away part of itself as holdout
    holdout_ids = pick_holdout(coco["images"], holdout_fraction, seed) if split == "train" else set()

    # Group the person boxes by image id
    persons_per_image = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotation"]:   # note: HIT-UAV uses "annotation", not "annotations"
        if ann["category_id"] == PERSON_ID:
            persons_per_image[ann["image_id"]].append(ann["bbox"])

    stats = {}   # out_split -> [images, person boxes, background images]
    for img in coco["images"]:
        name = img["filename"]
        out_split = "holdout" if img["id"] in holdout_ids else split

        img_dir = out_root / "images" / out_split
        lbl_dir = out_root / "labels" / out_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / "normal_json" / split / name, img_dir / name)

        lines = []
        for box in persons_per_image[img["id"]]:
            xc, yc, w, h = coco_box_to_yolo(box, img["width"], img["height"])
            lines.append(f"{PERSON_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # An empty .txt file means "no person here" (a background image)
        (lbl_dir / Path(name).with_suffix(".txt")).write_text("\n".join(lines))

        st = stats.setdefault(out_split, [0, 0, 0])
        st[0] += 1
        st[1] += len(lines)
        st[2] += (len(lines) == 0)

    for out_split, (n_img, n_box, n_bg) in stats.items():
        print(f"{out_split:7s}: {n_img} images, {n_box} person boxes, {n_bg} background images")


def write_data_yaml(out_root: Path):
    yaml_text = (
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/holdout   # used during training to pick best.pt (official val stays untouched)\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: person\n"
    )
    (out_root / "data.yaml").write_text(yaml_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/HIT-UAV", help="Folder of the downloaded HIT-UAV dataset")
    parser.add_argument("--out", default="data/processed/hit-uav-person", help="Where to write the YOLO dataset")
    parser.add_argument("--holdout", type=float, default=0.1, help="Fraction of train used as holdout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src_root, out_root = Path(args.src), Path(args.out)
    if not (src_root / "normal_json" / "annotations").is_dir():
        raise SystemExit(f"HIT-UAV not found in '{src_root}': expected a 'normal_json/annotations' folder inside it.")
    if out_root.exists():
        shutil.rmtree(out_root)   # start clean so no image ends up in two splits
    for split in SPLITS:
        convert_split(src_root, out_root, split, args.holdout, args.seed)
    write_data_yaml(out_root)
    print(f"Done. Dataset config: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
