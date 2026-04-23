#!/usr/bin/env python3
import argparse
import math
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}


def parse_voc_objects(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class '{name}' in {xml_path}")

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing <bndbox> in {xml_path}")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5

        bw_n = bw / width
        bh_n = bh / height
        major_n = max(bw_n, bh_n)
        minor_n = min(bw_n, bh_n)

        objects.append((CLASS_TO_ID[name], cx, cy, major_n, minor_n))

    return width, height, objects


def rotate_point(x: float, y: float, cx: float, cy: float, angle_deg: float):
    ang = math.radians(angle_deg)
    c = math.cos(ang)
    s = math.sin(ang)
    dx = x - cx
    dy = y - cy
    # PIL.Image.rotate(+ang) is CCW in image coordinates (x right, y down).
    xr = c * dx + s * dy + cx
    yr = -s * dx + c * dy + cy
    return xr, yr


def write_label(path: Path, items, image_w: int, image_h: int, theta_rad: float, angle_deg: float):
    lines = []
    icx = image_w * 0.5
    icy = image_h * 0.5
    dropped = 0

    for class_id, cx, cy, major_n, minor_n in items:
        if angle_deg != 0.0:
            cx, cy = rotate_point(cx, cy, icx, icy, angle_deg)

        cx_n = cx / image_w
        cy_n = cy / image_h
        if not (0.0 <= cx_n <= 1.0 and 0.0 <= cy_n <= 1.0):
            dropped += 1
            continue

        lines.append(
            f"{class_id} {cx_n:.6f} {cy_n:.6f} {major_n:.6f} {minor_n:.6f} {theta_rad:.8f}"
        )

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return dropped


def read_split_ids(path: Path):
    ids = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Robust to lines like: "2008_000001 1"
        ids.append(line.split()[0])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval-root", type=Path, default=Path("VOCdevkit/VOC2012"))
    parser.add_argument("--out-root", type=Path, default=Path("data/VOC2012_ellipse"))
    parser.add_argument("--max-angle", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-copy-images", action="store_true", help="Only write labels/splits, do not copy JPEG images")
    args = parser.parse_args()

    tv_img_dir = args.trainval_root / "JPEGImages"
    tv_ann_dir = args.trainval_root / "Annotations"
    tv_split_dir = args.trainval_root / "ImageSets" / "Main"

    if not tv_img_dir.is_dir() or not tv_ann_dir.is_dir() or not tv_split_dir.is_dir():
        raise FileNotFoundError(f"Train/val VOC directory structure is incomplete: {args.trainval_root}")

    out_images = args.out_root / "images"
    out_labels = args.out_root / "labels"
    out_splits = args.out_root / "splits"
    if not args.no_copy_images:
        out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)

    split_map = {}
    for split in ("train", "val", "trainval"):
        p = tv_split_dir / f"{split}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
        split_map[split] = read_split_ids(p)

    rng = random.Random(args.seed)
    base_ids = split_map["trainval"]
    if not base_ids:
        raise RuntimeError("trainval split is empty")

    dropped_labels = 0
    processed_base = 0
    generated_samples = 0
    copied_images = 0

    for image_id in tqdm(base_ids, desc="Converting trainval", unit="img"):
        image_path = tv_img_dir / f"{image_id}.jpg"
        xml_path = tv_ann_dir / f"{image_id}.xml"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {image_id}: {image_path}")
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing annotation for {image_id}: {xml_path}")

        width, height, objects = parse_voc_objects(xml_path)

        image = Image.open(image_path).convert("RGB")
        if image.size != (width, height):
            raise ValueError(f"Image size mismatch for {image_id}: xml=({width},{height}) img={image.size}")

        left_deg = rng.uniform(0.0, args.max_angle)
        right_deg = -rng.uniform(0.0, args.max_angle)
        aug_specs = [
            (f"{image_id}_orig", 0.0),
            (f"{image_id}_rotL", left_deg),
            (f"{image_id}_rotR", right_deg),
        ]

        for new_id, ang_deg in aug_specs:
            if not args.no_copy_images:
                rot = image.rotate(ang_deg, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
                rot.save(out_images / f"{new_id}.jpg", quality=95)
                copied_images += 1
            theta_rad = math.radians(ang_deg)
            dropped_labels += write_label(
                out_labels / f"{new_id}.txt",
                objects,
                width,
                height,
                theta_rad=theta_rad,
                angle_deg=ang_deg,
            )
            generated_samples += 1
        processed_base += 1

    expanded_splits = {}
    for split_name, ids in split_map.items():
        expanded = []
        for image_id in ids:
            expanded.append(f"{image_id}_orig")
            expanded.append(f"{image_id}_rotL")
            expanded.append(f"{image_id}_rotR")
        expanded_splits[split_name] = expanded
        (out_splits / f"{split_name}.txt").write_text("\n".join(expanded) + "\n", encoding="utf-8")

    (args.out_root / "classes.txt").write_text("\n".join(VOC_CLASSES) + "\n", encoding="utf-8")
    print(f"Processed base trainval images: {processed_base}")
    print(f"Generated samples: {generated_samples}")
    print(f"Copied images: {copied_images}")
    print(f"Dropped labels (center out of [0,1]): {dropped_labels}")
    print(f"Splits written: {', '.join(sorted(expanded_splits.keys()))}")
    print(f"Output root: {args.out_root}")


if __name__ == "__main__":
    main()
