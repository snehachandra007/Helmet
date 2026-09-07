"""Quick sanity check that a webcam is reachable.

Tries index 0 first, then 1 and 2 if that fails. Prints OK + frame shape
on success, or FAILED on miss. No model, no serial — just OpenCV.
"""

import sys

import cv2


def try_index(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame.shape


def main():
    for idx in (0, 1, 2):
        shape = try_index(idx)
        if shape is not None:
            print(f"OK  camera index {idx}  frame shape {shape}")
            return 0
        print(f"--  camera index {idx}  no frame")

    print("FAILED — no camera available at indices 0, 1, 2")
    return 1


if __name__ == "__main__":
    sys.exit(main())
