"""Five-second serial loopback — sends one JSON message per second
and prints anything coming back. Use this once the ESP32 echo/heartbeat
firmware is flashed but before you wire up the full perception loop.
"""

import json
import os
import sys
import time

import serial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def main():
    print(f"opening {config.SERIAL_PORT} @ {config.BAUD}")
    ser = serial.Serial(config.SERIAL_PORT, config.BAUD, timeout=0.2)
    time.sleep(2)  # ESP32 boot
    print("sending 1 msg/sec for 5 seconds")

    classes = ["helmet", "no_helmet", "helmet", "no_helmet", "helmet"]
    for i, cls in enumerate(classes):
        msg = {
            "ts": round(time.time(), 3),
            "class": cls,
            "conf": 0.85,
            "bbox": [120 + i * 5, 120, 240 + i * 5, 260],
            "frame_w": config.FRAME_W,
            "frame_h": config.FRAME_H,
        }
        line = json.dumps(msg) + "\n"
        ser.write(line.encode('utf-8'))
        print(f"-> {line.strip()}")

        t_end = time.time() + 1.0
        while time.time() < t_end:
            if ser.in_waiting:
                reply = ser.read(ser.in_waiting).decode('utf-8', errors='replace').strip()
                if reply:
                    print(f"<- {reply}")
            time.sleep(0.05)

    ser.close()
    print("done")


if __name__ == "__main__":
    main()
