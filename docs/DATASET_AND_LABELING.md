# Dataset preparation, cleaning, and annotation policy

## Canonical class order

The model has 60 classes, in this exact order:

1. Colors: `red`, `blue`, `green`, `yellow`, `orange`, `purple`.
2. Within each color: `cube`, `cuboid`, `sphere`, `cone`, `cylinder`.
3. For each shape: `small`, then `big`.

Class name format: `<color>_<shape>_<size>`. Class IDs are zero-based. **Never reorder names without remapping every label ID.**

## Size semantics

`small` and `big` are apparent image-scale labels, not real-world dimensions. Define a reproducible threshold using normalized bounding-box area or another explicit rule and apply it consistently. A box can appear small because it is farther from the camera, so the label is viewpoint-dependent. For robot picking, estimate physical size separately using calibrated depth/geometry; do not treat this classification label as metric dimensions.

## Expected directory layout

```text
root/
  images/train/ images/val/ images/test/
  labels/train/ labels/val/ labels/test/
```

Every image should have a same-relative-path `.txt` label file. Empty label files are allowed only when an image intentionally contains no target objects; investigate them in this synthetic shapes dataset.

## YOLO detection label format

One object per line:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0,1]` relative to image width/height. Width and height must be positive; the box must remain within image boundaries. This project is detection-only; segmentation masks and OBB labels use different formats and are not supported by the current audit/training scripts.

## Cleaning checklist

- Confirm the image opens and is not corrupted.
- Confirm one matching label file per image, with matching relative path/stem.
- Validate five values per annotation row; integer class ID `0..59`; finite normalized coordinates; positive width/height; box inside image.
- Find exact duplicates and check whether near-duplicates leak across splits. Exact SHA-256 checking is optional in the audit script; near-duplicate detection still requires review.
- Review class-instance counts and examples for every class.
- Visually inspect random images and overlays. Structural validation cannot catch incorrect class semantics or poorly fitted boxes.
- Ensure train/validation/test splits are independent. If images derive from the same scene/seed/render sequence, split by scene or source group rather than randomly splitting near-identical frames.
- Do not automatically delete or rewrite data. Back up data and make intentional, logged corrections.

## Acceptance gate

Do not train until the structural audit has no errors, all 60 class IDs match the canonical mapping, duplicate/leakage warnings have been reviewed, and representative annotations have been visually inspected. Report counts and known limitations honestly.
