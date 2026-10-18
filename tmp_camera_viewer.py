#!/usr/bin/env python3
"""Temporary OpenCV preview for a local camera."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import monotonic

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "index",
        nargs="?",
        type=int,
        default=1,
        help="OpenCV camera index (default: 1, usually the external camera)",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def sharpness_score(frame) -> float:
    """Return a simple focus score; compare it across the same scene and camera."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    return float(cv2.Laplacian(center, cv2.CV_64F).var())


def main() -> int:
    args = parse_args()
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.index, backend)

    if not camera.isOpened():
        print(
            f"Could not open camera index {args.index}. "
            "Try index 0, 1, or 2 and check macOS camera permissions.",
            file=sys.stderr,
        )
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = camera.get(cv2.CAP_PROP_FPS)
    window_name = f"Camera {args.index} — S to save, Q/Esc to close"
    output_path = Path(f"camera_{args.index}_sample.png").resolve()
    print(
        f"Viewing camera index {args.index} at {actual_width}x{actual_height}, "
        f"reported {reported_fps:.1f} FPS."
    )
    print("Press S to save a frame; press Q or Escape to close.")

    last_report = monotonic()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.", file=sys.stderr)
                return 1

            score = sharpness_score(frame)
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Center sharpness: {score:.1f}",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s"):
                if cv2.imwrite(str(output_path), frame):
                    print(f"Saved frame to {output_path}")
                else:
                    print(f"Could not save frame to {output_path}", file=sys.stderr)

            now = monotonic()
            if now - last_report >= 1.0:
                print(f"Sharpness score: {score:.1f}")
                last_report = now
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
