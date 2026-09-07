"""List every visible serial port, then send one dummy JSON message
to config.SERIAL_PORT and print whatever comes back within ~1 second.
"""

import json
import os
import sys
import time

import serial
import serial.tools.list_ports

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("(no serial ports detected)")
        return
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device:15}  {p.description}")


def main():
    list_ports()
    print()
    print(f"opening {config.SERIAL_PORT} @ {config.BAUD}")
    ser = serial.Serial(config.SERIAL_PORT, config.BAUD, timeout=1)
    time.sleep(2)  # ESP32 boot delay

    msg = {
        "ts": round(time.time(), 3),
        "class": "helmet",
        "conf": 0.99,
        "bbox": [100, 100, 200, 200],
        "frame_w": config.FRAME_W,
        "frame_h": config.FRAME_H,
    }
    line = json.dumps(msg) + "\n"
    print(f"-> {line.strip()}")
    ser.write(line.encode('utf-8'))

    time.sleep(1)
    if ser.in_waiting:
        reply = ser.read(ser.in_waiting).decode('utf-8', errors='replace').strip()
        print(f"<- {reply}")
    else:
        print("(no reply within 1s)")

    ser.close()


if __name__ == "__main__":
    main()
