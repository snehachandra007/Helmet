"""Event logger — appends one CSV row + one JPEG snapshot per logged event."""

import csv
import os
from datetime import datetime

import cv2


class EventLogger:
    HEADER = ['timestamp_iso', 'class', 'conf', 'bbox', 'snapshot']

    def __init__(self, csv_path, snapshot_dir):
        self.csv_path = csv_path
        self.snapshot_dir = snapshot_dir

        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)

        new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self._fh = open(csv_path, 'a', newline='', encoding='utf-8')
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow(self.HEADER)
            self._fh.flush()

    def log(self, frame, cls, conf, bbox):
        ts = datetime.now()
        ts_iso = ts.isoformat(timespec='milliseconds')
        snap_name = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_{cls}.jpg"
        snap_path = os.path.join(self.snapshot_dir, snap_name)

        if frame is not None:
            cv2.imwrite(snap_path, frame)

        bbox_str = ','.join(str(int(v)) for v in bbox) if bbox is not None else ''
        self._writer.writerow([ts_iso, cls, f"{conf:.3f}", bbox_str, snap_path])
        self._fh.flush()

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()
