from __future__ import annotations
import argparse, random
from pathlib import Path
import cv2
import yaml

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw YOLO boxes and class names for manual dataset QA.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("reports/label_samples"))
    args = ap.parse_args()
    cfg_path = args.data.expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = Path(cfg["path"]).expanduser().resolve()
    image_dir = Path(cfg.get(args.split, f"images/{args.split}"))
    if not image_dir.is_absolute(): image_dir = root / image_dir
    label_dir = root / "labels" / args.split
    names_raw = cfg.get("names", {})
    names = list(names_raw.values()) if isinstance(names_raw, dict) else list(names_raw)
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not images: raise SystemExit(f"No images found under {image_dir}")
    rng = random.Random(args.seed); selected = rng.sample(images, min(args.count, len(images)))
    out = args.output.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None: print(f"SKIP unreadable image: {image_path}"); continue
        relative = image_path.relative_to(image_dir).with_suffix(".txt")
        label_path = label_dir / relative
        h, w = image.shape[:2]
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5: continue
                try: cid = int(float(fields[0])); xc, yc, bw, bh = map(float, fields[1:])
                except ValueError: continue
                x1, y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
                x2, y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
                label = names[cid] if 0 <= cid < len(names) else f"INVALID_CLASS_{cid}"
                cv2.putText(image, label, (max(0,x1), max(18,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            cv2.putText(image, "MISSING LABEL FILE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        dest = out / f"{image_path.stem}_overlay.jpg"
        cv2.imwrite(str(dest), image)
    print(f"Wrote annotation overlays to {out}. Inspect class names and box alignment manually.")
    return 0


if __name__ == "__main__": raise SystemExit(main())
