# Claude Code Prompt — SafeRide Sentinel Perception Scaffold

Open Claude Code in an empty folder where you want the project to live. Then paste the prompt below. Claude Code will create the entire project structure and all the code files — you just plug in your trained `best.pt` later.

---

## How to use

1. Create an empty folder, e.g. `C:\Users\Ryan\projects\saferide-perception\`
2. Open a terminal there
3. Run `claude` (or `claude-code` depending on your install)
4. Paste the entire prompt below into the chat
5. Let it scaffold everything
6. Follow `INSTRUCTIONS.md` (which Claude Code will also create) to train your model on Colab and run the system

---

## The prompt to paste

```
You are scaffolding the perception subsystem for "SafeRide Sentinel," a robotics mini-project. The system uses a laptop webcam running YOLOv8 to detect motorcycle riders with/without helmets and sends JSON detections over USB serial to an ESP32 (built by a teammate) that controls pan-tilt servos and a barrier.

My role is the software/perception side only. My teammate handles the ESP32 firmware and hardware. I will train the YOLOv8 model myself on Google Colab and download `best.pt` into the `models/` folder.

Please build out the complete project in this folder with the following structure and files. Do not skip any. After scaffolding, give me a checklist of what to do next.

PROJECT STRUCTURE TO CREATE:

saferide-perception/
├── INSTRUCTIONS.md              # comprehensive step-by-step build guide
├── README.md                    # short summary for GitHub viewers
├── requirements.txt
├── config.py
├── perception.py                # main loop: webcam → YOLO → serial + logging
├── event_logger.py              # CSV + snapshot logger
├── .gitignore
├── tests/
│   ├── camera_test.py
│   ├── serial_test.py           # lists ports + sends one dummy JSON message
│   ├── inference_test.py        # runs YOLO on a static test image
│   └── loopback_test.py         # serial round-trip without full pipeline
├── dataset_prep/
│   ├── extract_frames.py        # CLI: video path, out dir, every-nth
│   └── README.md                # dataset assembly walkthrough (Kaggle, Roboflow, custom)
├── models/
│   └── .gitkeep                 # best.pt goes here, gitignored
├── logs/
│   ├── .gitkeep
│   └── snapshots/.gitkeep
└── docs/
    └── protocol.md              # JSON message contract, shared with firmware teammate

REQUIREMENTS FOR EACH FILE:

1) requirements.txt
   - ultralytics
   - opencv-python
   - pyserial
   - numpy
   Pin to recent stable versions if you know them; otherwise leave unpinned.

2) config.py — module-level constants:
   - SERIAL_PORT (default 'COM3', with comment about Mac/Linux paths)
   - BAUD = 115200
   - CAMERA_INDEX = 0
   - FRAME_W = 640, FRAME_H = 480
   - MODEL_PATH = 'models/best.pt'
   - CONF_THRESH = 0.5
   - TARGET_CLASSES = ('helmet', 'no_helmet')
   - SMOOTH_WINDOW = 5
   - SEND_INTERVAL = 0.15
   - LOG_CSV = 'logs/events.csv'
   - SNAPSHOT_DIR = 'logs/snapshots'

3) perception.py — main loop:
   - Load YOLO model from config.MODEL_PATH
   - Open serial port and time.sleep(2) after open (ESP32 auto-resets — critical)
   - Open webcam at config.CAMERA_INDEX with W/H from config
   - Per frame: run inference, pick highest-confidence detection in TARGET_CLASSES
   - Maintain a deque of recent class predictions for majority-vote smoothing (size SMOOTH_WINDOW)
   - Only send to serial if smoothed decision matches current best AND elapsed time > SEND_INTERVAL
   - JSON message format MUST be exactly: {"ts": float, "class": str, "conf": float, "bbox": [x1,y1,x2,y2], "frame_w": int, "frame_h": int}\n
   - Read any incoming bytes from serial (ESP32 heartbeats) and print them
   - Log a row + snapshot to event_logger only when the decision class changes (not every frame)
   - Show a live OpenCV preview window with annotated boxes; quit on 'q'
   - Wrap the main loop in try/finally that releases camera, closes window, closes serial, closes logger

4) event_logger.py:
   - Class EventLogger(csv_path, snapshot_dir)
   - Auto-create folders if missing
   - Write header row only if CSV is new: timestamp_iso, class, conf, bbox, snapshot
   - .log(frame, cls, conf, bbox) writes one row + saves a JPEG snapshot
   - .close() to flush and close the file handle

5) tests/camera_test.py:
   - Open VideoCapture(0), read one frame, print "OK" + shape or "FAILED"
   - Try indices 1 and 2 if 0 fails

6) tests/serial_test.py:
   - List all ports via serial.tools.list_ports
   - Then open config.SERIAL_PORT and send one dummy JSON message
   - Wait 1 second and print any reply

7) tests/inference_test.py:
   - Load model from config.MODEL_PATH
   - Run on a path passed as argv[1] (default to a placeholder)
   - Print every detection's class and confidence

8) tests/loopback_test.py:
   - Same as serial_test.py but in a 5-second loop, sending one message per second

9) dataset_prep/extract_frames.py:
   - argv[1]=video, argv[2]=out_dir, argv[3]=every_nth (default 10)
   - Save frames as frame_NNNNN.jpg
   - Print summary at end

10) docs/protocol.md:
    Document the perception → controller JSON shape exactly as in perception.py.
    Include: transport (USB serial, 115200, 8N1, line-delimited JSON \n), field table with types and notes, and an example controller→perception heartbeat shape: {"state": "ACTUATE", "pan": 92, "tilt": 48, "barrier": "DOWN"}

11) README.md (short, for GitHub viewers):
    - One-paragraph project description
    - "Software portion only" disclaimer
    - Quick start (4 commands: clone, venv, install, run)
    - Link to INSTRUCTIONS.md for the full guide
    - Link to docs/protocol.md for the firmware teammate

12) .gitignore — include:
    - saferide-env/ and venv/
    - __pycache__/ and *.pyc
    - models/*.pt and models/*.onnx
    - logs/events.csv, logs/snapshots/
    - .vscode/, .idea/, .DS_Store

13) INSTRUCTIONS.md — write a thorough step-by-step guide covering:
    - Prerequisites: Python 3.10/3.11, Git, venv setup commands for Windows + Mac/Linux
    - Installing dependencies
    - How to find the right COM port (using serial.tools.list_ports + Device Manager hint)
    - How to verify the camera works
    - Full dataset preparation walkthrough: Kaggle Helmet Detection link, Roboflow Universe, Roboflow Annotate workflow, the 3 classes (rider/helmet/no_helmet), why hard negatives (caps, turbans, hoods labeled no_helmet) matter, augmentation settings (flip horizontal, brightness ±25%, blur up to 1.5px, noise 1%), final dataset folder structure, zip and upload to Drive
    - Google Colab training: select T4 GPU, mount Drive, unzip dataset, train with these exact params: epochs=80, imgsz=640, batch=16, patience=15, augment=True, name='helmet_v1'
    - Validation interpretation: target ≥70% mAP@0.5, but specifically demand no_helmet recall ≥ 0.7 and precision ≥ 0.7 because false negatives let violators through and false positives shame innocent riders
    - Saving best.pt back to Drive, optional ONNX export
    - Bringing best.pt into models/ and running tests/inference_test.py
    - Test stages in strict order: camera → inference (no serial) → serial loopback → ESP32 integration
    - The full message protocol (point at docs/protocol.md)
    - Demo deliverables checklist from the project brief: GitHub repo, model weights, metrics report, 2-3 min demo video showing helmet pass + no-helmet block + pan-tilt tracking + one failure case, event log CSV, slide deck contributions
    - Troubleshooting table: SerialException + Arduino IDE port hold, missing first messages from no sleep(2), turbans classified as helmets, low FPS, black preview window, jittery servos, oscillating PID, ESP32 random resets from servo brownout, Mac OpenCV threading, ModuleNotFoundError missing venv
    - Week-by-week timeline matching the project brief: W1 dataset+train+protocol, W2 serial+smoothing+logging, W3 integration+latency, W4 demo+report

After creating all files, run `tree` (or equivalent ls -R) to show the final structure, and print a short next-steps checklist:
   1. python -m venv saferide-env, then activate
   2. pip install -r requirements.txt
   3. python tests/camera_test.py
   4. Train model on Colab per INSTRUCTIONS.md §4
   5. Drop best.pt into models/
   6. Coordinate with teammate on docs/protocol.md
   7. Run python perception.py once ESP32 is wired and ready

Do NOT install dependencies. Do NOT try to download the dataset. Just create the files exactly as specified.
```

---

## After Claude Code finishes scaffolding

You'll have a complete repo. Your manual work from there:

1. **Train the model** — follow `INSTRUCTIONS.md` §4 to train on Colab. Copy `best.pt` into `models/` when done.
2. **Configure for your machine** — edit `config.py`: change `SERIAL_PORT` to your actual COM port (find it with `python tests/serial_test.py`), change `CAMERA_INDEX` if 0 doesn't work.
3. **Share `docs/protocol.md` with your friend** — they implement the ESP32 side against the same spec.
4. **Test in stages** — camera → inference → serial → full integration. Don't skip.

If anything goes wrong during scaffolding (Claude Code skips a file, gets a path wrong), just ask it: "you missed `tests/loopback_test.py` — please create it per the spec" and it'll fix it.

---

## Tip: keep the prompt and the instructions file in your repo

Once Claude Code has built the project, copy `INSTRUCTIONS.md` and this `CLAUDE_CODE_PROMPT.md` into the repo root and commit them. That way:
- Anyone (including your teammate) can re-scaffold the project from scratch
- Your evaluator sees how you organized the build
- If you start over for the final-year version, you have your prompt history