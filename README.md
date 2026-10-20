# YOLO26 Shapes Detection — End-to-End Project

A reproducible, beginner-friendly workflow for auditing a YOLO-format dataset, preparing a machine-specific dataset config, training YOLO26, evaluating on a held-out test set, running image/video inference, and testing a live camera. The repository contains code and documentation only; do **not** commit the 18,000-image dataset, model weights, runs, or secrets.

## Project scope

- 60 classes: 6 colors × 5 shapes × 2 apparent-size labels.
- Colors: red, blue, green, yellow, orange, purple.
- Shapes: cube, cuboid, sphere, cone, cylinder.
- Size suffix: `small` / `big` describes **apparent size in the image**, not physical dimensions. Dataset naming currently uses `<color>_<shape>_<size>` (for example `red_cube_small`). Keep this mapping fixed across training and future Isaac Sim data.
- Dataset split expected: train 14,400; val 1,800; test 1,800 (18,000 total). The audit checks your actual files rather than assuming these numbers prove quality.

## 1. Clone and create an environment (Linux VM)

```bash
git clone https://github.com/aravindkumar67/Datasetyoloobjectdetection.git
cd Datasetyoloobjectdetection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build compatible with the VM's NVIDIA driver first. For the Dev4 VM previously inspected (NVIDIA A10, driver supporting CUDA 12.8), try the CUDA 12.8 wheels:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

If the wheel version is unavailable for your platform or the CUDA check fails, consult the official PyTorch selector for a compatible build; do not upgrade a managed VM driver without admin approval. CPU mode is possible but much slower.

## 2. Point the project to your dataset

The dataset should already be extracted and retain the YOLO layout:

```text
DATASET_ROOT/
  images/{train,val,test}/...
  labels/{train,val,test}/...
  data.yaml                 # original may contain a path from another machine
```

For the VM path discussed during setup:

```bash
python scripts/prepare_dataset.py \
  --dataset-root "$HOME/yolo-detection/yolo_shapes_v2_18000_images/yolo_shapes_v2_18000" \
  --output configs/data_dev4.yaml
```

This creates a **separate** config with an absolute path and runs a non-destructive audit. It does not rename classes, alter labels, delete files, or silently repair annotations. Read the report and fix any errors before training.

Expected class order is color-major, then shape order `cube, cuboid, sphere, cone, cylinder`, each with `small` then `big`. The script checks for 60 names and label IDs in range. If the actual `data.yaml` class names/order differ, stop and resolve the mapping rather than training with mismatched IDs.

## 3. Audit the dataset

```bash
python scripts/audit_dataset.py --data configs/data_dev4.yaml --report reports/dataset_audit.json
```

The audit checks split folders, image readability, image/label pairing, empty/malformed labels, class IDs, normalized bounding boxes, duplicate image hashes, and split leakage. It writes a JSON report. Duplicate images and split leakage are warnings that require review; the script will not delete data automatically.

Review examples manually too. Automated checks cannot confirm that a box tightly encloses the correct object, that a shape/color is labeled correctly, or that the `small`/`big` boundary is meaningful. Dataset size categories are viewpoint-dependent; document a consistent labeling rule before expanding the dataset.

## 4. Check the model and run a smoke test

```bash
python scripts/check_environment.py
python scripts/train.py --data configs/data_dev4.yaml --model yolo26n.pt --epochs 2 --batch 8 --imgsz 640 --device 0 --name smoke_test
```

The first run may download pretrained weights. If `yolo26n.pt` is unavailable for your installed Ultralytics version, stop and verify the supported YOLO26 model identifier in the official Ultralytics documentation. Do not silently switch model families.

After the smoke test works, run a longer experiment:

```bash
python scripts/train.py --data configs/data_dev4.yaml --model yolo26n.pt --epochs 100 --batch -1 --imgsz 640 --device 0 --name shapes_yolo26n
```

`--batch -1` asks Ultralytics to estimate a suitable batch size on the GPU. If memory errors occur, use `--batch 8` or `--batch 4`. Keep experiment names unique. Default output is under `runs/detect/`.

## 5. Evaluate the held-out test split

Use the `best.pt` produced by training. Do not use the test set to repeatedly tune hyperparameters.

```bash
python scripts/evaluate.py --data configs/data_dev4.yaml --weights runs/detect/shapes_yolo26n/weights/best.pt --split test --device 0
```

Review precision, recall, mAP50, mAP50-95, per-class results, and confusion matrix. A high aggregate score can hide weak classes. Do not claim 99.99% accuracy unless a clearly defined metric and independent evaluation substantiate it.

## 6. Inference

Single image/folder:

```bash
python scripts/predict.py --weights runs/detect/shapes_yolo26n/weights/best.pt --source path/to/image_or_folder --conf 0.25 --save
```

Video file:

```bash
python scripts/predict.py --weights runs/detect/shapes_yolo26n/weights/best.pt --source path/to/video.mp4 --conf 0.25 --save
```

Live camera (try index 0 first):

```bash
python scripts/camera.py --weights runs/detect/shapes_yolo26n/weights/best.pt --source 0 --conf 0.25
```

A camera attached to your laptop is not automatically available inside a remote VM. For a remote VM, use a supported camera stream or run camera inference on the machine that physically owns the camera. OpenCV camera support depends on the VM/session configuration. Press `q` to exit the live window.

## 7. Output and reproducibility

- Dataset report: `reports/dataset_audit.json`
- Training artifacts: `runs/detect/<experiment-name>/`
- Validation/test metrics and plots: Ultralytics output directory
- Best checkpoint: `runs/detect/<experiment-name>/weights/best.pt`

Record the dataset release/version, class mapping, code commit, Ultralytics/PyTorch versions, model name, image size, confidence threshold, and evaluation metrics with every experiment. Keep model weights and generated reports local unless you intentionally publish them.

## Pipeline

1. Environment and GPU check.
2. Dataset path/config preparation.
3. Read-only data cleaning/audit and manual visual review.
4. Two-epoch smoke training.
5. Full YOLO26 training.
6. Validation during training; final held-out test evaluation.
7. Image/video inference.
8. Live camera inference.
9. Later: Isaac Sim live RGB integration and separate depth/3D perception modules. This repository currently implements 2D object detection, not tracking, segmentation, OBB, depth estimation, robot grasp planning, or robot control.

## Troubleshooting

- `CUDA available: False`: check driver/PyTorch CUDA build compatibility with `python scripts/check_environment.py`.
- Dataset path missing: rerun `prepare_dataset.py` with the exact extracted dataset root.
- Labels missing or invalid: inspect JSON report and correct source data intentionally; no automatic deletion/repair occurs.
- Camera cannot open: try `--source 0`, check camera access/permissions, or use a stream/locally connected camera.
- Out of memory: lower batch size or image size.

## Safety and scope

This is an offline perception/training project. It does not send motion commands to a physical robot. Any later robot integration must include coordinate-frame validation, reachability/collision checks, speed limits, emergency stop, and hardware-in-the-loop testing.
