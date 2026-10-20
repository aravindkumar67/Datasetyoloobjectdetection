from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a YOLO26 detector using a verified dataset YAML.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="yolo26n.pt", help="YOLO26 checkpoint/model identifier supported by installed Ultralytics")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics estimate batch size")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="0 for first GPU; cpu for CPU")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="shapes_yolo26n")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not args.data.expanduser().is_file(): ap.error(f"Dataset YAML not found: {args.data}")
    if args.epochs < 1 or args.imgsz < 32: ap.error("epochs must be positive and imgsz >= 32")
    model = YOLO(args.model)
    results = model.train(data=str(args.data.expanduser().resolve()), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device, workers=args.workers, name=args.name, project=args.project, patience=args.patience, seed=args.seed, deterministic=True, plots=True, save=True, resume=args.resume)
    print(f"Training finished. Results: {getattr(results, 'save_dir', Path(args.project) / args.name)}")
    print("Use the resulting weights/best.pt for held-out test evaluation and inference.")
    return 0


if __name__ == "__main__": raise SystemExit(main())
