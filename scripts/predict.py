from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser(description="Run YOLO detection on an image, directory, video, or stream URL.")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, help="Image/folder/video path, camera index, or stream URL")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save", action="store_true", help="Save annotated results")
    ap.add_argument("--show", action="store_true", help="Display results; requires a GUI")
    ap.add_argument("--project", default="runs/predict")
    ap.add_argument("--name", default="shapes")
    args = ap.parse_args()
    if not args.weights.expanduser().is_file(): ap.error(f"Weights not found: {args.weights}")
    if not 0 <= args.conf <= 1: ap.error("conf must be between 0 and 1")
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(str(args.weights.expanduser().resolve()))
    results = model.predict(source=source, conf=args.conf, imgsz=args.imgsz, device=args.device, save=args.save, show=args.show, project=args.project, name=args.name, stream=False)
    print(f"Processed {len(results)} image/frame result(s).")
    if args.save: print(f"Annotated output is under {args.project}/{args.name}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
