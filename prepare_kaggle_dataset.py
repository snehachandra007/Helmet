"""
Convert Kaggle 'andrewmvd/helmet-detection' (Pascal VOC) to YOLOv8 format
with a 80/20 train/valid split.

Usage:
    python prepare_kaggle_dataset.py <input_dir> <output_dir>
Example:
    python prepare_kaggle_dataset.py ./helmet-detection ./helmet_dataset
"""
import os, sys, glob, shutil, random
import xml.etree.ElementTree as ET

# Class names IN ORDER — index = YOLO class ID
CLASSES = ['With Helmet', 'Without Helmet']
# What we want them renamed to in data.yaml (so they match your perception code)
EXPORT_NAMES = ['helmet', 'no_helmet']

SPLIT = 0.8   # 80% train, 20% valid


def convert_box(size, box):
    """VOC (xmin,ymin,xmax,ymax) → YOLO (x_center,y_center,w,h) normalized."""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh


def convert_one(xml_path, out_label_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    lines = []
    for obj in root.iter('object'):
        cls = obj.find('name').text.strip()
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        b = obj.find('bndbox')
        box = (float(b.find('xmin').text), float(b.find('ymin').text),
               float(b.find('xmax').text), float(b.find('ymax').text))
        bb = convert_box((w, h), box)
        lines.append(f"{cls_id} {' '.join(f'{v:.6f}' for v in bb)}")

    with open(out_label_path, 'w') as f:
        f.write('\n'.join(lines))


def main(in_dir, out_dir):
    img_dir = os.path.join(in_dir, 'images')
    ann_dir = os.path.join(in_dir, 'annotations')

    # Collect pairs
    pairs = []
    for img in glob.glob(os.path.join(img_dir, '*.*')):
        base = os.path.splitext(os.path.basename(img))[0]
        xml = os.path.join(ann_dir, base + '.xml')
        if os.path.exists(xml):
            pairs.append((img, xml, base))
    print(f"Found {len(pairs)} image/annotation pairs.")

    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * SPLIT)
    train, valid = pairs[:split_idx], pairs[split_idx:]
    print(f"Train: {len(train)}  Valid: {len(valid)}")

    for split_name, items in [('train', train), ('valid', valid)]:
        img_out = os.path.join(out_dir, split_name, 'images')
        lbl_out = os.path.join(out_dir, split_name, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img, xml, base in items:
            shutil.copy(img, os.path.join(img_out, os.path.basename(img)))
            convert_one(xml, os.path.join(lbl_out, base + '.txt'))

    # data.yaml
    yaml = (
        f"path: {os.path.abspath(out_dir)}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"nc: {len(EXPORT_NAMES)}\n"
        f"names: {EXPORT_NAMES}\n"
    )
    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        f.write(yaml)

    print("\nDone. Folder structure:")
    print(f"  {out_dir}/data.yaml")
    print(f"  {out_dir}/train/images, train/labels")
    print(f"  {out_dir}/valid/images, valid/labels")
    print("\nNext: zip the output folder and upload to Google Drive.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])