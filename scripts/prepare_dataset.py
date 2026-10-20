from __future__ import annotations
import argparse
from pathlib import Path
import yaml

EXPECTED_CLASSES = [
    f"{color}_{shape}_{size}"
    for color in ("red", "blue", "green", "yellow", "orange", "purple")
    for shape in ("cube", "cuboid", "sphere", "cone", "cylinder")
    for size in ("small", "big")
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local YOLO data YAML without modifying source labels.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Extracted dataset root containing images/ and labels/.")
    parser.add_argument("--output", default="configs/data_local.yaml", type=Path, help="Output YAML path.")
    parser.add_argument("--source-yaml", type=Path, help="Optional original data.yaml to read class names from.")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset root does not exist: {root}")
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            path = root / kind / split
            if not path.is_dir():
                parser.error(f"Required directory missing: {path}")
    names = EXPECTED_CLASSES
    if args.source_yaml:
        source_path = args.source_yaml.expanduser().resolve()
        if not source_path.is_file():
            parser.error(f"Source YAML not found: {source_path}")
        source = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        source_names = source.get("names")
        if isinstance(source_names, dict):
            names = [source_names[k] if k in source_names else source_names[str(k)] for k in range(len(source_names))]
        elif isinstance(source_names, list):
            names = source_names
        else:
            parser.error("Source YAML must define names as a list or numeric-keyed mapping.")
    if len(names) != 60:
        parser.error(f"Expected 60 class names; found {len(names)}. Review class mapping before training.")
    if list(names) != EXPECTED_CLASSES:
        parser.error("Class order differs from the project's documented 60-class mapping. Do not train until IDs are reconciled.")
    payload = {"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test", "names": {i: name for i, name in enumerate(names)}}
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Created config: {out}")
    print(f"Dataset root: {root}")
    print("Class mapping verified: 60 expected classes")
    print("Next: python scripts/audit_dataset.py --data " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
