# SafeRide Sentinel — Software Build Instructions (Your Portion)

Your scope: laptop camera → YOLOv8 helmet detection → JSON over USB serial → ESP32 (your friend's part).

This document walks through every step from empty laptop to working perception node integrated with your friend's firmware. Follow in order. Don't skip the test stages — debugging an integrated system is 10x harder than debugging each piece alone.

---

## Table of Contents

1. Prerequisites & laptop setup
2. Project structure
3. Dataset preparation
4. Train the model on Google Colab
5. Bring the trained model back to the laptop
6. Serial protocol contract (lock this with your friend in week 1)
7. Test stages (camera → inference → serial loopback → ESP32)
8. Run the integrated system
9. Demo prep & deliverables checklist
10. Troubleshooting cheat sheet

---

## 1. Prerequisites & laptop setup

### 1.1 Install Python 3.10 or 3.11

Don't use 3.12 — Ultralytics has occasional compatibility issues. Download from python.org. During install, check **"Add Python to PATH"**.

Verify:

```bash
python --version
# Should print: Python 3.10.x or 3.11.x
```

### 1.2 Install Git

From git-scm.com. You'll need it for version control and to share work with your team.

### 1.3 Create a virtual environment

Open a terminal in your project folder:

```bash
python -m venv saferide-env
```

Activate it (do this every time you open a new terminal):

```bash
# Windows (PowerShell)
.\saferide-env\Scripts\Activate.ps1

# Windows (cmd)
saferide-env\Scripts\activate.bat

# Mac / Linux
source saferide-env/bin/activate
```

You'll see `(saferide-env)` at the start of your prompt when active.

### 1.4 Install dependencies

```bash
pip install --upgrade pip
pip install ultralytics opencv-python pyserial numpy
```

This pulls ~2 GB (PyTorch is included with Ultralytics). Be patient.

### 1.5 Verify webcam works

Save this as `tests/camera_test.py` and run it:

```python
import cv2

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print("Camera index 0:", "OK" if ok else "FAILED",
      "shape:", frame.shape if ok else None)
cap.release()
```

If it fails, try indices 1 or 2 — laptops with IR cameras (Windows Hello) sometimes register multiple camera devices.

### 1.6 Find your serial port name

Plug in the ESP32 to USB. Then:

```python
from serial.tools import list_ports
for p in list_ports.comports():
    print(p.device, "-", p.description)
```

Look for `Silicon Labs CP210x USB to UART Bridge` or `CH340` — that's your ESP32. Common names:
- Windows: `COM3`, `COM4`, `COM5`, ...
- Mac: `/dev/cu.SLAB_USBtoUART` or `/dev/cu.usbserial-XXXX`
- Linux: `/dev/ttyUSB0` or `/dev/ttyACM0`

Save this name — you'll use it in `config.py`.

---

## 2. Project structure

Create this folder layout in your project root:

```
saferide-perception/
├── INSTRUCTIONS.md              # this file
├── README.md                    # short summary for GitHub
├── requirements.txt             # pinned dependencies
├── config.py                    # serial port, thresholds, paths
├── perception.py                # main loop: camera → YOLO → serial
├── event_logger.py              # CSV logging of every detection
├── tests/
│   ├── camera_test.py           # webcam smoke test
│   ├── serial_test.py           # list ports, open & send dummy JSON
│   ├── inference_test.py        # YOLO on a static image
│   └── loopback_test.py         # full pipeline with serial echo
├── dataset_prep/
│   ├── extract_frames.py        # pull every Nth frame from a video
│   └── README.md                # dataset assembly steps
├── models/
│   └── best.pt                  # downloaded from Colab (gitignored)
├── logs/
│   ├── events.csv               # one row per detection event
│   └── snapshots/               # JPEG saved per detection (gitignored)
├── docs/
│   └── protocol.md              # JSON message schema (shared with firmware)
└── .gitignore
```

`.gitignore` should include `saferide-env/`, `models/*.pt`, `logs/`, `__pycache__/`, `.vscode/`.

---

## 3. Dataset preparation

The brief recommends 3 classes: `rider`, `helmet`, `no_helmet`. Stick with that — it makes the system much more robust.

### 3.1 Source images

Combine these for ~2,000–3,000 labeled images:

1. **Kaggle Helmet Detection dataset** — kaggle.com/datasets/andrewmvd/helmet-detection (~764 annotated images). Download, unzip.
2. **Roboflow Universe** — universe.roboflow.com → search "motorcycle helmet" → pick a dataset with the same class labels you're using → download as YOLOv8 format.
3. **Self-collected video** — record 5–10 minutes at your campus gate or a busy junction (with permission). Phone camera at chest height, landscape orientation.

### 3.2 Extract frames from your video

Save as `dataset_prep/extract_frames.py`:

```python
"""Extract every Nth frame from a video for labeling."""
import cv2, os, sys

video_path = sys.argv[1]
out_dir = sys.argv[2]
every_nth = int(sys.argv[3]) if len(sys.argv) > 3 else 10

os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)

count, saved = 0, 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if count % every_nth == 0:
        path = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
    count += 1

cap.release()
print(f"Saved {saved} frames from {count} total.")
```

Run it:

```bash
python dataset_prep/extract_frames.py gate_video.mp4 raw_frames/ 10
```

### 3.3 Label your custom frames

Use **Roboflow Annotate** (free for hobby use):

1. Create a project on roboflow.com → "Object Detection"
2. Upload your raw frames
3. Define classes: `rider`, `helmet`, `no_helmet`
4. Draw bounding boxes. Aim for ~150–200 labeled images (one labeling session = 2–3 hours).
5. **Important — hard negatives**: deliberately label heads with caps, turbans, hoods, beanies as `no_helmet`. Without these, your model says "helmet" for any head covering.

### 3.4 Merge datasets in Roboflow

In Roboflow, upload the Kaggle and Universe datasets too (or merge them locally). Make sure class names match exactly across all three sources. Roboflow has a "remap classes" feature for this.

### 3.5 Apply augmentation in Roboflow

Configure these under "Generate":
- **Auto-Orient**: yes
- **Resize**: 640×640
- **Flip**: horizontal (no vertical — riders aren't usually upside down)
- **Brightness**: ±25%
- **Blur**: up to 1.5px (simulates motion blur)
- **Noise**: up to 1%

Generate → "Export" → choose **YOLOv8** format → "show download code" → copy the snippet.

### 3.6 Final dataset structure

After export, you'll have:

```
helmet_dataset/
├── data.yaml          # class names + paths
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

Zip the whole `helmet_dataset/` folder and upload to Google Drive — you'll need it in Colab.

---

## 4. Train the model on Google Colab

Free Colab gives you a T4 GPU which trains YOLOv8n in ~30–45 minutes.

### 4.1 Create a new Colab notebook

colab.research.google.com → New Notebook. Runtime → Change runtime type → **T4 GPU**.

### 4.2 Cell 1 — install and mount Drive

```python
!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')
```

### 4.3 Cell 2 — unzip dataset

```python
!unzip -q /content/drive/MyDrive/helmet_dataset.zip -d /content/
```

Verify the path to `data.yaml`:

```python
!cat /content/helmet_dataset/data.yaml
```

If paths inside `data.yaml` are wrong, fix them to be absolute (`/content/helmet_dataset/train/images`, etc.).

### 4.4 Cell 3 — train

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # n = nano, fast on CPU later

results = model.train(
    data='/content/helmet_dataset/data.yaml',
    epochs=80,
    imgsz=640,
    batch=16,
    patience=15,
    augment=True,
    name='helmet_v1',
    project='/content/runs'
)
```

Watch the metrics. By epoch 30–40 you should see mAP@0.5 climbing past 0.6. If it's stuck below 0.4 by epoch 50, your dataset is too small or labels are bad — stop and fix.

### 4.5 Cell 4 — validate and review

```python
metrics = model.val()
print("mAP@0.5:", metrics.box.map50)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)

# Per-class breakdown
for i, name in enumerate(model.names.values()):
    print(f"  {name}: P={metrics.box.p[i]:.3f}  R={metrics.box.r[i]:.3f}  mAP={metrics.box.maps[i]:.3f}")
```

**Acceptance bar before moving on**: `no_helmet` recall ≥ 0.7 and precision ≥ 0.7. If recall is low, you're letting violators through. If precision is low, you'll falsely flag helmeted riders. Either way, collect more data and retrain.

### 4.6 Cell 5 — save the model

```python
!cp /content/runs/helmet_v1/weights/best.pt /content/drive/MyDrive/best.pt
!cp -r /content/runs/helmet_v1 /content/drive/MyDrive/helmet_v1_results
```

Optional — export to ONNX for ~2x faster CPU inference:

```python
model.export(format='onnx', imgsz=640)
!cp /content/runs/helmet_v1/weights/best.onnx /content/drive/MyDrive/best.onnx
```

### 4.7 Test predictions on a sample image

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('/content/runs/helmet_v1/weights/best.pt')
results = model('/content/helmet_dataset/valid/images/some_image.jpg')

for r in results:
    plt.imshow(r.plot()[:,:,::-1])
    plt.axis('off')
    plt.show()
```

If predictions look reasonable, you're done with Colab.

---

## 5. Bring the trained model back to the laptop

Download `best.pt` from Drive into `models/best.pt` in your project. Don't commit it — it's typically 6–12 MB and gitignored.

Test inference locally:

```python
# tests/inference_test.py
from ultralytics import YOLO
import cv2

model = YOLO('models/best.pt')
img = cv2.imread('tests/sample.jpg')   # any test image
results = model(img)[0]

for box in results.boxes:
    cls = model.names[int(box.cls)]
    conf = float(box.conf)
    print(f"{cls}: {conf:.2f}")
```

Expect 8–15 FPS on laptop CPU at 640px. If you have an NVIDIA GPU and CUDA installed, it'll be 30+ FPS.

---

## 6. Serial protocol contract

**Lock this in week 1 with your friend.** Both sides implement against the same JSON shape — don't change field names without telling them.

Save as `docs/protocol.md`:

```markdown
# Serial Message Protocol

## Transport
- USB serial, 115200 baud, 8N1
- Line-delimited JSON (one object per line, terminated with `\n`)

## Perception → Controller (laptop → ESP32)

```json
{"ts": 1714500000.123, "class": "no_helmet", "conf": 0.87,
 "bbox": [320, 180, 420, 320], "frame_w": 640, "frame_h": 480}
```

| Field      | Type    | Notes                                          |
|------------|---------|------------------------------------------------|
| ts         | float   | Unix timestamp, seconds                        |
| class      | string  | "helmet" or "no_helmet" only                   |
| conf       | float   | Detection confidence, 0.0–1.0                  |
| bbox       | int[4]  | [x1, y1, x2, y2] in pixel coords               |
| frame_w    | int     | Source frame width in pixels                   |
| frame_h    | int     | Source frame height in pixels                  |

## Controller → Perception (ESP32 → laptop, optional)

```json
{"state": "ACTUATE", "pan": 92, "tilt": 48, "barrier": "DOWN"}
```

Used for debugging and the perception side's logs. Not required for the system to function.
```

---

## 7. Test stages

Don't skip these. Each isolates one failure mode.

### Stage 1 — camera only

Run `tests/camera_test.py`. Confirm webcam opens at index 0 (or whichever index works).

### Stage 2 — inference only (no serial)

Comment out the `ser.write(...)` line in `perception.py`. Run it. Wear a helmet, take it off in front of the camera. Watch the live preview window — bounding boxes should track your head, classifications flip between `helmet` and `no_helmet`.

### Stage 3 — serial without ESP32 (loopback test)

Save as `tests/loopback_test.py`. Skip the actual ESP32 — just open the port and verify writes don't error:

```python
import serial, json, time
from config import SERIAL_PORT, BAUD

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
time.sleep(2)   # ESP32 reboot grace

msg = {"ts": time.time(), "class": "helmet", "conf": 0.95,
       "bbox": [100, 100, 200, 200], "frame_w": 640, "frame_h": 480}
ser.write((json.dumps(msg) + '\n').encode())
print("Sent:", msg)

time.sleep(1)
if ser.in_waiting:
    print("ESP32 reply:", ser.readline().decode(errors='ignore').strip())

ser.close()
```

If the ESP32 has the loopback firmware (just echoes back what it receives), you'll see the message printed back. If not, you'll just confirm the port opens without errors.

### Stage 4 — full integration with your friend's firmware

Run `perception.py`. Watch:
- Live preview window opens
- Pan/tilt servos track your head as you move side to side
- Barrier opens (if `helmet`) or stays closed + buzzer (if `no_helmet`)

Measure end-to-end latency: hold a helmet up, time how long until the barrier responds. Target < 1.5s. If it's slower:
- Lower `imgsz` in inference to 480
- Increase `SEND_INTERVAL` (less work per second, more responsive per send)
- Check if your friend's firmware has long `delay()` calls

---

## 8. Run the integrated system

### 8.1 `config.py`

```python
"""Shared configuration. Edit these for your machine."""

# Serial
SERIAL_PORT = 'COM3'        # change to your port (Windows: COMx, Mac/Linux: /dev/...)
BAUD = 115200

# Camera
CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480

# Inference
MODEL_PATH = 'models/best.pt'
CONF_THRESH = 0.5
TARGET_CLASSES = ('helmet', 'no_helmet')

# Smoothing & rate limiting
SMOOTH_WINDOW = 5           # majority vote across last N frames
SEND_INTERVAL = 0.15        # seconds between serial writes (~6.7 Hz)

# Logging
LOG_CSV = 'logs/events.csv'
SNAPSHOT_DIR = 'logs/snapshots'
```

### 8.2 `event_logger.py`

```python
"""CSV logger for detection events — one of the deliverables."""
import csv, os, cv2
from datetime import datetime

class EventLogger:
    def __init__(self, csv_path, snapshot_dir):
        self.csv_path = csv_path
        self.snapshot_dir = snapshot_dir
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)
        new = not os.path.exists(csv_path)
        self.fp = open(csv_path, 'a', newline='', encoding='utf-8')
        self.writer = csv.writer(self.fp)
        if new:
            self.writer.writerow(['timestamp_iso', 'class', 'conf',
                                  'bbox', 'snapshot'])

    def log(self, frame, cls, conf, bbox):
        ts = datetime.now().isoformat(timespec='seconds')
        snap = os.path.join(self.snapshot_dir,
                            f"{ts.replace(':','-')}_{cls}.jpg")
        cv2.imwrite(snap, frame)
        self.writer.writerow([ts, cls, f"{conf:.3f}", str(bbox), snap])
        self.fp.flush()

    def close(self):
        self.fp.close()
```

### 8.3 `perception.py`

```python
"""Main perception loop — webcam, YOLO, serial, logging."""
import cv2, json, time
from collections import deque
import serial
from ultralytics import YOLO

import config
from event_logger import EventLogger

def main():
    model = YOLO(config.MODEL_PATH)
    ser = serial.Serial(config.SERIAL_PORT, config.BAUD, timeout=0.1)
    time.sleep(2)   # ESP32 resets when port opens — let it boot

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)

    logger = EventLogger(config.LOG_CSV, config.SNAPSHOT_DIR)
    recent = deque(maxlen=config.SMOOTH_WINDOW)
    last_send = 0
    last_logged_class = None

    print("SafeRide perception running. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed. Stopping.")
                break

            h, w = frame.shape[:2]
            results = model(frame, verbose=False, conf=config.CONF_THRESH)[0]

            best = None
            for box in results.boxes:
                cls = model.names[int(box.cls)]
                if cls not in config.TARGET_CLASSES:
                    continue
                conf = float(box.conf)
                if best is None or conf > best['conf']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    best = {'class': cls, 'conf': conf,
                            'bbox': [x1, y1, x2, y2]}

            recent.append(best['class'] if best else None)
            votes = [c for c in recent if c]
            decision = max(set(votes), key=votes.count) if votes else None

            now = time.time()
            if (best and decision == best['class']
                    and now - last_send > config.SEND_INTERVAL):
                msg = {'ts': now, 'class': best['class'],
                       'conf': best['conf'], 'bbox': best['bbox'],
                       'frame_w': w, 'frame_h': h}
                ser.write((json.dumps(msg) + '\n').encode())
                last_send = now

                if decision != last_logged_class:
                    logger.log(frame, best['class'], best['conf'],
                               best['bbox'])
                    last_logged_class = decision

            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    print("ESP32:", line)

            cv2.imshow('SafeRide Sentinel', results.plot())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        logger.close()
        print("Shutdown clean.")

if __name__ == '__main__':
    main()
```

Run with:

```bash
python perception.py
```

---

## 9. Demo prep & deliverables checklist

From section 12 of the brief, your software-side deliverables:

- [ ] GitHub repo with perception code, README, requirements.txt
- [ ] Trained model weights `best.pt` shared via Drive (too big for git)
- [ ] Metrics report — copy `helmet_v1_results/` from Colab into `docs/metrics/`
- [ ] Demo video — 2–3 min showing helmet pass, no-helmet block, edge cases
- [ ] Event log CSV from `logs/events.csv` after the demo run
- [ ] Slide deck contributions: architecture diagram, dataset stats, mAP/precision/recall numbers, latency measurements

For the demo video specifically, capture:
1. Approach with helmet → barrier opens, green LED
2. Approach without helmet → buzzer + red LED, barrier stays
3. Pan-tilt actively tracking lateral movement (this is the closed-loop differentiator vs. a pure CV project)
4. One failure case + your explanation of why (examiners love honesty)

---

## 10. Troubleshooting cheat sheet

| Symptom | Cause | Fix |
|---|---|---|
| `serial.SerialException: could not open port 'COM3'` | Arduino IDE Serial Monitor is open, holding the port exclusively | Close the Serial Monitor before running Python |
| First few detections don't reach ESP32 | Auto-reset on serial open eats the first messages | The `time.sleep(2)` after opening is mandatory |
| YOLO classifies turbans/caps as helmets | Insufficient hard negatives | Add 50+ images of caps/turbans/hoods labeled `no_helmet`, retrain |
| Inference < 5 FPS | CPU bottleneck at 640px | Lower `imgsz` to 480, or export to ONNX |
| Live preview window is black | Camera index wrong | Try indices 1 or 2 in `config.CAMERA_INDEX` |
| Servos jitter even when nothing's in frame | No temporal smoothing on detection class | The deque-based majority vote in perception.py prevents this |
| Pan-tilt overshoots and oscillates | PID gains too high (your friend's side) | Tell them: lower Kp, add small Kd |
| ESP32 randomly resets | Servos drawing power from ESP32's 5V | Hardware fix: separate 5V 3A supply for servos with shared ground |
| `cv2.imshow` window doesn't open on Mac | Mac threading issue with OpenCV | Run from terminal, not Jupyter |
| `ModuleNotFoundError: ultralytics` | Forgot to activate venv | `saferide-env\Scripts\activate` first |

---

## Key dates from the 4-week plan

| Week | Your software focus |
|---|---|
| 1 | Dataset assembly, train v1, hit ≥ 70% mAP, lock the JSON protocol |
| 2 | Serial integration with loopback Arduino, smoothing, event logging |
| 3 | Full integration with hardware, latency tuning |
| 4 | Demo video, metrics report, slide deck |

Don't let dataset work bleed into week 3. If you don't have a working model by end of week 1, you cut scope (drop the `rider` class, ship with 2 classes) and move on.

---

*Built for SafeRide Sentinel mini-project. Last updated: May 2026.*