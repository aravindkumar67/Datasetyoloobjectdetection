from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
from PIL import Image
import yaml

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def files_under(folder: Path, suffixes: set[str]) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only YOLO dataset audit. Never deletes or rewrites dataset files.")
    ap.add_argument("--data", required=True, type=Path, help="Machine-specific YOLO data YAML")
    ap.add_argument("--report", type=Path, default=Path("reports/dataset_audit.json"))
    ap.add_argument("--hash-duplicates", action="store_true", help="Hash image contents to detect exact duplicates across splits (slower).")
    args = ap.parse_args()
    cfg_path = args.data.expanduser().resolve()
    if not cfg_path.is_file(): ap.error(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    root = Path(cfg.get("path", cfg_path.parent)).expanduser()
    if not root.is_absolute(): root = (cfg_path.parent / root).resolve()
    names_raw = cfg.get("names", {})
    names = list(names_raw.values()) if isinstance(names_raw, dict) else list(names_raw)
    issues, warnings, split_summary = [], [], {}
    global_hashes: dict[str, str] = {}
    class_counts = Counter()
    for split in ("train", "val", "test"):
        image_dir = root / cfg.get(split, f"images/{split}")
        if not image_dir.is_absolute(): image_dir = root / image_dir
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            issues.append(f"{split}: image directory missing: {image_dir}"); continue
        if not label_dir.is_dir():
            issues.append(f"{split}: label directory missing: {label_dir}"); continue
        images = files_under(image_dir, EXTS)
        labels = files_under(label_dir, {".txt"})
        image_stems = {p.relative_to(image_dir).with_suffix("").as_posix(): p for p in images}
        label_stems = {p.relative_to(label_dir).with_suffix("").as_posix(): p for p in labels}
        missing_labels = sorted(set(image_stems) - set(label_stems))
        orphan_labels = sorted(set(label_stems) - set(image_stems))
        if missing_labels: issues.append(f"{split}: {len(missing_labels)} images have no label file; examples: {missing_labels[:5]}")
        if orphan_labels: warnings.append(f"{split}: {len(orphan_labels)} label files have no matching image; examples: {orphan_labels[:5]}")
        bad_images, empty_labels, bad_rows = [], [], []
        for stem, image_path in image_stems.items():
            try:
                with Image.open(image_path) as im:
                    im.verify()
                if args.hash_duplicates:
                    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                    if digest in global_hashes:
                        warnings.append(f"Exact duplicate image across/within splits: {image_path} matches {global_hashes[digest]}")
                    else: global_hashes[digest] = str(image_path)
            except Exception as exc:
                bad_images.append(f"{image_path}: {exc}")
            label_path = label_stems.get(stem)
            if not label_path: continue
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                bad_rows.append(f"{label_path}: cannot read: {exc}"); continue
            if not lines or not any(line.strip() for line in lines):
                empty_labels.append(str(label_path)); continue
            for line_no, line in enumerate(lines, 1):
                if not line.strip(): continue
                fields = line.split()
                if len(fields) != 5:
                    bad_rows.append(f"{label_path}:{line_no}: expected 5 values, found {len(fields)}"); continue
                try:
                    class_float = float(fields[0]); coords = [float(x) for x in fields[1:]]
                    if not class_float.is_integer() or not (0 <= int(class_float) < len(names)):
                        raise ValueError(f"class id must be integer in [0,{len(names)-1}]")
                    if not all(math.isfinite(x) for x in coords): raise ValueError("non-finite coordinate")
                    x, y, w, h = coords
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("bbox values out of normalized range or non-positive width/height")
                    if x - w/2 < -1e-6 or y - h/2 < -1e-6 or x + w/2 > 1+1e-6 or y + h/2 > 1+1e-6:
                        raise ValueError("bbox extends outside image bounds")
                    class_counts[int(class_float)] += 1
                except Exception as exc:
                    bad_rows.append(f"{label_path}:{line_no}: {exc} | {line}")
        if bad_images: issues.append(f"{split}: {len(bad_images)} unreadable images; examples: {bad_images[:5]}")
        if empty_labels: warnings.append(f"{split}: {len(empty_labels)} empty label files; examples: {empty_labels[:5]}")
        if bad_rows: issues.append(f"{split}: {len(bad_rows)} invalid label rows; examples: {bad_rows[:10]}")
        split_summary[split] = {"images": len(images), "labels": len(labels), "missing_label_files": len(missing_labels), "orphan_label_files": len(orphan_labels), "unreadable_images": len(bad_images), "empty_label_files": len(empty_labels), "invalid_label_rows": len(bad_rows)}
    class_summary = {str(i): {"name": names[i], "instances": class_counts.get(i, 0)} for i in range(len(names))}
    unused = [v["name"] for v in class_summary.values() if v["instances"] == 0]
    if unused: warnings.append(f"Classes with zero labeled instances: {unused}")
    report = {"dataset_root": str(root.resolve()), "config": str(cfg_path), "class_count": len(names), "split_summary": split_summary, "class_instances_by_id": class_summary, "issues": issues, "warnings": warnings, "passed_structural_checks": not issues}
    out = args.report.expanduser().resolve(); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved: {out}")
    if issues: print("AUDIT FAILED: resolve issues before training."); return 2
    print("STRUCTURAL AUDIT PASSED. Manually inspect sample annotations before training.")
    return 0


if __name__ == "__main__": raise SystemExit(main())
