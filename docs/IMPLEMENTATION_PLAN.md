# Implementation Plan — YOLO26 Shapes Detection

## Objective
Build a repeatable YOLO26 object-detection workflow for a 60-class synthetic shapes dataset, then validate inference and prepare a safe path toward Isaac Sim integration.

## Fixed dataset semantics
- 6 colors: red, blue, green, yellow, orange, purple.
- 5 shapes: cube, cuboid, sphere, cone, cylinder.
- 2 apparent image-size labels: small, big.
- 60 total classes; IDs must match `configs/data.example.yaml` exactly.
- Dataset release expected to contain 18,000 images split 14,400/1,800/1,800. The audit must verify the actual VM copy.

## Phases and acceptance gates

### Phase 0 — Environment
1. Activate an isolated Python virtual environment.
2. Install PyTorch compatible with the NVIDIA driver before Ultralytics.
3. Run `python scripts/check_environment.py`.
4. Acceptance: imports work and intended GPU is visible, or CPU mode is explicitly accepted.

### Phase 1 — Dataset configuration
1. Keep extracted dataset separate from this code repository.
2. Generate a VM-specific YAML using `scripts/prepare_dataset.py`.
3. Verify the 60 class names and class IDs against the canonical list.
4. Acceptance: image/label directories exist and class mapping is exact.

### Phase 2 — Cleaning and QA
1. Run `scripts/audit_dataset.py`.
2. Review missing/orphan labels, corrupt images, invalid rows, empty labels, per-class counts and duplicate warnings.
3. Run `scripts/visualize_labels.py` and inspect box alignment/class labels manually.
4. Confirm no near-duplicate leakage across train/validation/test; exact hashes alone cannot detect every near duplicate.
5. Fix issues intentionally, preserving backups and documenting changes. Scripts do not automatically delete or rewrite data.
6. Acceptance: no structural errors and manual visual review completed.

### Phase 3 — Baseline training
1. Train a 2-epoch smoke test with YOLO26 nano.
2. Confirm the model loads, dataset resolves, loss decreases, and checkpoint/plots are written.
3. If smoke test passes, run the longer experiment with unique run name and saved settings.
4. Acceptance: training completes and `best.pt` exists.

### Phase 4 — Evaluation
1. Use validation metrics during iteration.
2. Evaluate the final candidate once on the held-out test split.
3. Review precision, recall, mAP50, mAP50-95, per-class metrics, and confusion matrix.
4. Acceptance: report measured metrics, not unsupported accuracy claims; investigate weak classes.

### Phase 5 — Inference
1. Run `scripts/predict.py` on sample images and videos.
2. Run `scripts/camera.py` where a camera is accessible.
3. Run `scripts/track.py` with ByteTrack for persistent track IDs.
4. Acceptance: visual results are saved/reviewed and camera-source constraints are understood.

### Phase 6 — Additional perception tasks
Detection labels cannot train segmentation or OBB models. If needed, prepare separately annotated datasets in the correct format, create task-specific YAMLs/checkpoints, train and evaluate those tasks independently. Do not reuse detection boxes as if they were masks or oriented boxes. The generic inference path can load compatible task-specific Ultralytics weights, but this repository does not yet include a validated segmentation/OBB dataset or model.

### Phase 7 — Isaac Sim integration
1. Keep the camera static relative to the conveyor/robot base.
2. Start with stationary objects, then enable moving conveyor.
3. Add RGB frame IDs/timestamps and live YOLO inference overlay.
4. Integrate ByteTrack IDs and target lifecycle/motion estimates.
5. Use Isaac depth and calibrated camera intrinsics/extrinsics to compute 3D coordinates.
6. Validate transforms against known world points before planning grasps.
7. Add reachability, collision, target-age, stale-frame, lost-track, and emergency-stop checks before robot execution.
8. Acceptance: simulation-only validation passes before any physical robot connection.

## Reproducibility
Record dataset release, code commit, Python/PyTorch/Ultralytics versions, model identifier, image size, batch size, seed, run name, and evaluation metrics. Keep raw data, weights, and run outputs outside Git unless deliberately publishing them.

## Known limitations
- Synthetic-only training data may not generalize to real camera imagery.
- `small`/`big` is apparent image size, not physical size.
- Current code implements detection, tracking, audit, training, evaluation and inference. It does not yet implement depth estimation, calibration, segmentation/OBB dataset preparation, grasp planning, or robot control.
- No target metric such as 99.99% is guaranteed; only measured evaluation results can support claims.
