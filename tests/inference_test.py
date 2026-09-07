"""Run YOLOv8 once on a static image and print every detection.

Usage:  python tests/inference_test.py path/to/image.jpg
If no path is given, falls back to a placeholder so you can sanity-check
that the model file at config.MODEL_PATH at least loads.
"""

import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def main():
    if not os.path.exists(config.MODEL_PATH):
        print(f"FAILED — model not found at {config.MODEL_PATH}")
        print("Train on Colab per INSTRUCTIONS.md §4 and copy best.pt into models/")
        return 1

    img_path = sys.argv[1] if len(sys.argv) > 1 else "tests/sample.jpg"

    print(f"loading model from {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)

    if not os.path.exists(img_path):
        print(f"(no test image at {img_path}) — model loaded OK, skipping inference")
        print("Pass an image path:  python tests/inference_test.py path/to/img.jpg")
        return 0

    print(f"running on {img_path}")
    results = model.predict(img_path, conf=config.CONF_THRESH, verbose=False)
    r = results[0]
    names = model.names

    if r.boxes is None or len(r.boxes) == 0:
        print("no detections")
        return 0

    print(f"detections: {len(r.boxes)}")
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = [int(v) for v in box.xyxy[0].tolist()]
        print(f"  [{i}] {names.get(cls_id, cls_id):12} conf={conf:.3f}  bbox={bbox}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
