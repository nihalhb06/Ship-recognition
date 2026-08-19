"""
Real-time ship detection from a camera feed using a trained YOLOv8 model.

Usage:
    python webcam_detect.py
    python webcam_detect.py --weights runs/train/ship_yolov8s/weights/best.pt --camera 0
    python webcam_detect.py --source path/to/video.mp4        # run on a video file instead
    python webcam_detect.py --save output.mp4                 # save the annotated feed

Press 'q' to quit the display window.
"""

import argparse
import time

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Real-time ship detection")
    p.add_argument("--weights", type=str, default="runs/train/ship_yolov8s/weights/best.pt",
                   help="path to trained model weights (best.pt)")
    p.add_argument("--camera", type=int, default=0, help="camera index (0 = default webcam)")
    p.add_argument("--source", type=str, default=None,
                   help="use a video file / RTSP stream instead of a camera")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", type=str, default=None,
                   help="cuda device id (e.g. 0) or 'cpu'; auto-detected if omitted")
    p.add_argument("--save", type=str, default=None, help="optional path to save annotated video")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")
    model = YOLO(args.weights)

    source = args.source if args.source is not None else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    prev_time = time.time()
    print("Running - press 'q' in the window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream ended or camera read failed.")
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )
        annotated = results[0].plot()

        now = time.time()
        fps_display = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Ship Detection (press q to quit)", annotated)

        if writer is not None:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
