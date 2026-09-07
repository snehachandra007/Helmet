# SafeRide Sentinel — Perception

Laptop-side perception subsystem for SafeRide Sentinel: a webcam runs
YOLOv8 to detect motorcycle riders with and without helmets, and streams
JSON detections over USB serial to an ESP32 that drives a pan-tilt rig
and a barrier.

> **Software portion only.** This repo contains the Python perception
> code, the dataset preparation tools, and the message protocol. The
> ESP32 firmware, servo wiring, and barrier hardware are built by my
> teammate against [docs/protocol.md](docs/protocol.md).

## Quick start

```bash
git clone <this repo>
cd saferide-perception
python -m venv saferide-env && saferide-env\Scripts\activate    # Windows
# source saferide-env/bin/activate                              # macOS / Linux
pip install -r requirements.txt
python perception.py
```

You will need [models/best.pt](models/) before `perception.py` does
anything useful — train it on Colab per the guide below.

## Where to go next

- **Full build guide:** [INSTRUCTIONS.md](INSTRUCTIONS.md) — environment,
  dataset, Colab training, validation targets, integration testing,
  troubleshooting, week-by-week plan.
- **Firmware contract:** [docs/protocol.md](docs/protocol.md) — share
  this with whoever writes the ESP32 side.
- **Dataset assembly:** [dataset_prep/README.md](dataset_prep/README.md).
