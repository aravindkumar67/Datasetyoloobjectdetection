from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained detector on val or held-out test split.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/evaluation")
    ap.add_argument("--name", default="test_results")
    args = ap.parse_args()
    for path in (args.data, args.weights):
        if not path.expanduser().is_file(): ap.error(f"File not found: {path}")
    model = YOLO(str(args.weights.expanduser().resolve()))
    metrics = model.val(data=str(args.data.expanduser().resolve()), split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device, plots=True, project=args.project, name=args.name, exist_ok=True)
    print("Evaluation complete. Inspect per-class metrics, confusion matrix, and saved plots.")
    for key in ("map", "map50", "map75", "mp", "mr"):
        value = getattr(metrics.box, key, None)
        if value is not None: print(f"{key}: {float(value):.6f}")
    print(f"Results directory: {getattr(metrics, 'save_dir', args.project)}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
