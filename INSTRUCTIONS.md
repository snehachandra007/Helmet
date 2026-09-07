# SafeRide Sentinel — Build Guide (Software / Perception)

This is the full step-by-step. Read it once before you start touching
anything; then come back and follow it section by section.

## Contents

1. [Prerequisites & environment](#1-prerequisites--environment)
2. [Find the right COM port and verify the camera](#2-find-the-right-com-port-and-verify-the-camera)
3. [Dataset preparation](#3-dataset-preparation)
4. [Training on Google Colab](#4-training-on-google-colab)
5. [Validating the model](#5-validating-the-model)
6. [Bringing the model into the project](#6-bringing-the-model-into-the-project)
7. [Test order — do not skip](#7-test-order--do-not-skip)
8. [The message protocol](#8-the-message-protocol)
9. [Demo deliverables](#9-demo-deliverables)
10. [Troubleshooting](#10-troubleshooting)
11. [Week-by-week timeline](#11-week-by-week-timeline)

---

## 1. Prerequisites & environment

You need:

- **Python 3.10 or 3.11.** (3.12 mostly works but Ultralytics + OpenCV
  wheels are most reliable on 3.10/3.11.)
- **Git.**
- A working **webcam**.
- A USB cable and an ESP32 board (your teammate's) when you reach the
  integration stage.

### Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv saferide-env
saferide-env\Scripts\Activate.ps1
```

If activation is blocked, run once as admin:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

**macOS / Linux:**

```bash
python3 -m venv saferide-env
source saferide-env/bin/activate
```

You should now see `(saferide-env)` prefixing your prompt.

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This pulls Ultralytics (YOLOv8), OpenCV, pyserial, NumPy, and their
transitive deps. First install is ~1.5 GB and takes a while.

---

## 2. Find the right COM port and verify the camera

### Camera

```bash
python tests/camera_test.py
```

You should see `OK  camera index 0  frame shape (480, 640, 3)`. If
index 0 fails the script also tries 1 and 2; whichever one works,
update `CAMERA_INDEX` in [config.py](config.py).

### Serial port

Plug in the ESP32 (or any USB-serial adapter) and run:

```bash
python tests/serial_test.py
```

The first thing it prints is every visible serial port. On Windows
ESP32s usually appear as `COM3`/`COM4`/.. — cross-check with **Device
Manager → Ports (COM & LPT)**. On macOS look for
`/dev/tty.usbserial-XXXX` or `/dev/tty.SLAB_USBtoUART`. On Linux it's
typically `/dev/ttyUSB0` or `/dev/ttyACM0`.

Edit `SERIAL_PORT` in [config.py](config.py) to match.

If `serial_test.py` then opens that port and sends one dummy JSON line,
you're good. Don't worry yet if there's no reply — that needs the
firmware flashed.

---

## 3. Dataset preparation

Goal: ≥ ~1500 labeled images split train/valid/test, with 3 classes:

- `rider` — the person on the bike
- `helmet` — clearly enclosed by a real helmet
- `no_helmet` — bare head **plus hard negatives**

### Sources (use all three)

1. **Kaggle Helmet Detection** datasets — search "helmet detection" on
   Kaggle. Fastest start. Verify a sample of labels by hand; some are
   noisy.
2. **Roboflow Universe** (`https://universe.roboflow.com`) — many
   pre-labeled YOLO-format helmet datasets you can fork.
3. **Your own clips** — phone footage of your local roads. Drop into
   `dataset_raw/` and run:

   ```bash
   python dataset_prep/extract_frames.py my_clip.mp4 dataset_raw/clip1 10
   ```

### Why hard negatives matter

Without them the model treats `cap == helmet` because both round things
sit on heads. Include caps, beanies, turbans, hoodies, headscarves, and
construction hats in `no_helmet`. Aim for ~15–20% of `no_helmet` to be
hard negatives.

This isn't pedantry — false negatives let actual violators through, and
false positives shame innocent riders. Both kinds of error are visible
in your demo.

### Labeling workflow (Roboflow Annotate)

1. New project → set classes to exactly `rider`, `helmet`, `no_helmet`.
   Spelling matters; this is what the model emits.
2. Upload images.
3. Label with bounding boxes. Use keyboard shortcuts; do it in one or
   two sittings.
4. Generate a version. On the version, enable augmentations:

   | Augmentation | Setting          |
   | ------------ | ---------------- |
   | Flip         | Horizontal       |
   | Brightness   | ±25%             |
   | Blur         | up to 1.5 px     |
   | Noise        | up to 1% pixels  |

5. Export → **YOLOv8** format → download zip.

The export looks like:

```
dataset/
├── data.yaml
├── train/  images/  labels/
├── valid/  images/  labels/
└── test/   images/  labels/
```

Zip the `dataset/` folder. Upload the zip to Google Drive into a folder
like `MyDrive/saferide/dataset.zip`.

---

## 4. Training on Google Colab

1. Open https://colab.research.google.com → New notebook.
2. **Runtime → Change runtime type → T4 GPU.** Confirm with `!nvidia-smi`.
3. Mount Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Unzip the dataset:

   ```python
   !mkdir -p /content/dataset
   !unzip -q /content/drive/MyDrive/saferide/dataset.zip -d /content/dataset
   !ls /content/dataset
   ```

5. Install Ultralytics:

   ```python
   !pip install -q ultralytics
   ```

6. Train. Use these exact parameters as a starting point:

   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')   # nano — runs on a laptop CPU at ~15 FPS
   results = model.train(
       data='/content/dataset/data.yaml',
       epochs=80,
       imgsz=640,
       batch=16,
       patience=15,
       augment=True,
       name='helmet_v1',
   )
   ```

   `yolov8s.pt` is the safer accuracy bump if your laptop has a GPU.

7. After training, the best weights live at
   `runs/detect/helmet_v1/weights/best.pt`. Copy back to Drive:

   ```python
   !cp runs/detect/helmet_v1/weights/best.pt /content/drive/MyDrive/saferide/best.pt
   ```

8. (Optional) Export to ONNX for faster CPU inference:

   ```python
   YOLO('runs/detect/helmet_v1/weights/best.pt').export(format='onnx')
   ```

---

## 5. Validating the model

Look at `results.png` and `confusion_matrix.png` in
`runs/detect/helmet_v1/`. The headline number is **mAP@0.5 ≥ 0.70**.

But that single number hides the thing that actually matters. Demand:

- **`no_helmet` recall ≥ 0.7** — false negatives let violators through.
- **`no_helmet` precision ≥ 0.7** — false positives shame innocent riders.

If `no_helmet` recall is low, you don't have enough `no_helmet` images
or your annotations are inconsistent. If precision is low, you don't
have enough hard negatives — go back to §3 and add them.

Don't move on until both numbers clear 0.7.

---

## 6. Bringing the model into the project

1. Download `best.pt` from Drive.
2. Drop it into `models/best.pt` in this repo (the path
   [config.py](config.py) `MODEL_PATH` points at).
3. Smoke test it loads:

   ```bash
   python tests/inference_test.py path/to/some_test_image.jpg
   ```

   You should see one detection per visible head, with class and
   confidence printed.

`models/*.pt` is gitignored — the weights file is too big and often
private. Keep a copy in Drive as your backup.

---

## 7. Test order — do not skip

Each stage isolates one failure mode. If you skip ahead and something
breaks, you won't know which layer is at fault.

| # | Test                                  | What it proves                       |
| - | ------------------------------------- | ------------------------------------ |
| 1 | `python tests/camera_test.py`         | OpenCV can grab frames from the cam. |
| 2 | `python tests/inference_test.py img`  | Trained model loads and detects.     |
| 3 | `python tests/serial_test.py`         | Host can open the serial port at all.|
| 4 | `python tests/loopback_test.py`       | ESP32 receives line-delimited JSON.  |
| 5 | `python perception.py`                | End-to-end: cam → YOLO → ESP32.      |

Stage 4 needs your teammate's firmware to at least echo or print
incoming lines. Stage 5 needs the full firmware including servos.

---

## 8. The message protocol

The wire format is locked in [docs/protocol.md](docs/protocol.md).
Share that file with the firmware teammate; both sides MUST match.

In short: **line-delimited UTF-8 JSON, one object per line, terminated
by `\n`**, over USB serial at 115200 8N1. Host sleeps 2s after opening
the port (ESP32 auto-resets on DTR). Detection messages have keys
`ts`, `class`, `conf`, `bbox`, `frame_w`, `frame_h`.

---

## 9. Demo deliverables

From the project brief, you owe:

- [ ] **GitHub repo** — this directory, pushed (without `models/*.pt`).
- [ ] **Model weights** — `best.pt` in your Drive, link in the report.
- [ ] **Metrics report** — screenshots of `results.png` and the
      confusion matrix, plus the per-class precision/recall numbers.
- [ ] **2–3 minute demo video** showing all of:
   - helmet rider passes (barrier stays up)
   - no-helmet rider blocked (barrier drops)
   - pan-tilt camera tracking a moving target
   - one honest failure case (and your read on why)
- [ ] **Event log CSV** — `logs/events.csv` from the demo run, with
      snapshots referenced.
- [ ] **Slide deck contributions** — your perception section.

---

## 10. Troubleshooting

| Symptom                                                        | Likely cause                                                    | Fix                                                                                       |
| -------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `serial.SerialException: could not open port`                  | Arduino IDE / another program is holding the port.              | Close the Serial Monitor in Arduino IDE. Only one process can own a COM port.             |
| First few JSON messages never reach the ESP32.                 | Missing 2-second sleep after `Serial(...)`.                     | The DTR toggle on open auto-resets the chip. Always `time.sleep(2)` after open.           |
| Turbans / caps classified as helmets.                          | Not enough hard negatives in training data.                     | Add 100+ `no_helmet` images of caps/turbans/hoodies. Retrain.                             |
| Low FPS (<5) on laptop.                                        | Using `yolov8s` or larger on CPU; or HD frames.                 | Use `yolov8n`, drop `imgsz` to 416, keep frame at 640×480, close other apps.              |
| Black preview window.                                          | Camera index wrong, or another app has the camera.              | Run `tests/camera_test.py`; close Zoom/Teams/OBS; try indices 1 and 2.                    |
| Jittery servos.                                                | Sending detections faster than the ESP32 can move servos.       | `SEND_INTERVAL` already throttles to 150ms. Raise it if needed; smooth on the ESP32 too.  |
| Oscillating PID on pan-tilt.                                   | Gains too high; or no deadband.                                 | Lower `Kp`, add a deadband of a few pixels around frame center. Firmware-side fix.        |
| ESP32 random resets when servos engage.                        | Servo brownout — servos drawing more than USB can supply.       | Power servos from a separate 5V/2A supply with a common ground; do not run them off USB.  |
| `cv2.imshow` hangs or fails on macOS.                          | OpenCV threading issue with the system Python on macOS.         | Use the venv's Python, not system Python; install `opencv-python` (not `headless`).       |
| `ModuleNotFoundError: No module named 'ultralytics'` etc.      | Forgot to activate the venv.                                    | `saferide-env\Scripts\activate` (Win) or `source saferide-env/bin/activate` (mac/linux).  |
| `cv2.error: !_src.empty()` on inference.                       | Image path is wrong or the file failed to load.                 | Pass an absolute path; verify with `python -c "import cv2; print(cv2.imread('x').shape)"`.|

---

## 11. Week-by-week timeline

This matches the project brief. Slip means cutting scope, not
extending; protect the demo week.

| Week | Goals                                                                                  |
| ---- | -------------------------------------------------------------------------------------- |
| W1   | Dataset assembled and labeled. First Colab training run finished. Protocol spec agreed with firmware teammate. |
| W2   | Serial path working end to end (loopback). Smoothing logic in. CSV + snapshot logging in. |
| W3   | Full integration with ESP32 + servos. Latency measured (target: < 200 ms decision-to-actuation). One honest failure case identified. |
| W4   | Demo video recorded. Metrics report written. Slides done. Repo cleaned up and pushed.  |
