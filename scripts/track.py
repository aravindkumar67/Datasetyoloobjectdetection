from __future__ import annotations
import argparse
import cv2
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser(description="Run persistent multi-object tracking on video, stream, or live camera.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="Video path, stream URL, or camera index such as 0")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config, e.g. bytetrack.yaml or botsort.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--project", default="runs/track")
    ap.add_argument("--name", default="shapes_bytetrack")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    results = model.track(source=source, tracker=args.tracker, conf=args.conf, imgsz=args.imgsz, device=args.device, show=args.show, save=args.save, persist=True, stream=False, project=args.project, name=args.name, verbose=False)
    print(f"Tracking processed {len(results)} frame result(s).")
    if args.save: print(f"Saved outputs under {args.project}/{args.name}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
