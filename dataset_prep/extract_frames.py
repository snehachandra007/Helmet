"""Extract frames from a video for dataset assembly.

Usage:
    python dataset_prep/extract_frames.py <video> <out_dir> [every_nth=10]

Saves frames as frame_NNNNN.jpg (5-digit zero-padded) into out_dir.
Prints a one-line summary at the end.
"""

import os
import sys

import cv2


def main():
    if len(sys.argv) < 3:
        print("usage: extract_frames.py <video> <out_dir> [every_nth=10]")
        return 1

    video_path = sys.argv[1]
    out_dir = sys.argv[2]
    every_nth = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    if not os.path.exists(video_path):
        print(f"video not found: {video_path}")
        return 1

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"could not open video: {video_path}")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every_nth == 0:
            name = f"frame_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(
        f"done — read {frame_idx} frames ({total} reported, {fps:.1f} fps), "
        f"saved {saved} every-{every_nth} into {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
