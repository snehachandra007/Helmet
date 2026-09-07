"""SafeRide Sentinel — perception main loop.

Webcam → YOLOv8 → JSON over USB serial to the ESP32 controller.
Also logs decision-change events to CSV with snapshots, and prints
any heartbeat/state bytes coming back from the controller.

Run:  python perception.py
Quit: press 'q' in the preview window.
"""

import json
import time
from collections import Counter, deque

import cv2
import serial
from ultralytics import YOLO

import config
from event_logger import EventLogger


def majority_vote(window):
    if not window:
        return None
    cls, _ = Counter(window).most_common(1)[0]
    return cls


def main():
    print(f"[init] loading model from {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)
    class_names = model.names  # {id: name}

    print(f"[init] opening serial {config.SERIAL_PORT} @ {config.BAUD}")
    ser = serial.Serial(config.SERIAL_PORT, config.BAUD, timeout=0.05)
    # CRITICAL: ESP32 auto-resets when the USB-serial DTR line toggles on open.
    # Without this sleep, the first JSON messages are lost while the chip boots.
    time.sleep(2)
    print("[init] serial ready")

    print(f"[init] opening camera index {config.CAMERA_INDEX}")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera at index {config.CAMERA_INDEX}")

    logger = EventLogger(config.LOG_CSV, config.SNAPSHOT_DIR)

    smooth = deque(maxlen=config.SMOOTH_WINDOW)
    last_sent_ts = 0.0
    last_logged_class = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] frame grab failed")
                continue

            results = model.predict(frame, conf=config.CONF_THRESH, verbose=False)
            r = results[0]

            best_cls = None
            best_conf = 0.0
            best_bbox = None

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    if cls_name not in config.TARGET_CLASSES:
                        continue
                    if conf > best_conf:
                        best_conf = conf
                        best_cls = cls_name
                        best_bbox = [float(v) for v in box.xyxy[0].tolist()]

            if best_cls is not None:
                smooth.append(best_cls)

            decision = majority_vote(list(smooth))

            now = time.time()
            if (
                decision is not None
                and best_cls is not None
                and decision == best_cls
                and (now - last_sent_ts) > config.SEND_INTERVAL
            ):
                msg = {
                    "ts": round(now, 3),
                    "class": best_cls,
                    "conf": round(best_conf, 3),
                    "bbox": [int(v) for v in best_bbox],
                    "frame_w": config.FRAME_W,
                    "frame_h": config.FRAME_H,
                }
                line = json.dumps(msg) + "\n"
                ser.write(line.encode('utf-8'))
                last_sent_ts = now

            # Log only on decision-class change.
            if decision is not None and decision != last_logged_class:
                logger.log(frame, decision, best_conf, best_bbox)
                last_logged_class = decision

            # Drain any incoming bytes from the controller.
            if ser.in_waiting:
                try:
                    incoming = ser.read(ser.in_waiting).decode('utf-8', errors='replace').strip()
                    if incoming:
                        print(f"[esp32] {incoming}")
                except Exception as e:
                    print(f"[esp32 read error] {e}")

            annotated = r.plot() if hasattr(r, 'plot') else frame
            label = decision if decision else 'none'
            cv2.putText(
                annotated,
                f"smoothed: {label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("SafeRide Sentinel — perception", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("[shutdown] releasing resources")
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        try:
            logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
