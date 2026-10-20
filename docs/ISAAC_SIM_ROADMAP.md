# Roadmap: YOLO perception toward Isaac Sim pick-and-place

This document records the intended architecture and separates implemented functionality from future work. The current code supports 2D detection, training/evaluation, image/video inference, and optional ByteTrack tracking. It does not yet implement a tested Isaac Sim extension or robot-control system.

## Target pipeline

1. **Simulation layer:** Isaac Sim scene, UR10E, gripper, conveyor, objects, fixed camera.
2. **Sensor layer:** synchronized RGB, Isaac depth, camera intrinsics/extrinsics, timestamp and frame ID.
3. **2D perception:** YOLO26 detection first; later separate task-specific models for segmentation and oriented bounding boxes (OBB), each with its own annotation format and evaluation.
4. **Tracking:** ByteTrack associates detections across frames. Track ID is temporal identity, not a permanent physical-object identity. Add lost-track handling and association validation.
5. **Depth validation:** compare depth estimated from RGB against Isaac ground-truth depth only as an experiment. The current detector does not estimate depth.
6. **3D perception:** use segmentation mask + valid depth + intrinsics to compute camera-frame XYZ, then apply calibrated camera-to-world and world-to-robot transforms. Check units and coordinate handedness.
7. **Geometry and targets:** estimate object dimensions/orientation; define a fixed conveyor pick zone; lock a target; estimate conveyor motion from timestamps and track history.
8. **Grasp planning:** generate candidate grasps, check gripper constraints/clearance/reachability, and produce a TCP target pose.
9. **Execution safety:** validate collision-free motion, stop conditions, stale-frame timeout, object disappearance, dropped tracks, and emergency-stop behavior before any robot motion.

## Camera and conveyor decisions

- Camera is fixed relative to the conveyor/robot base, not wrist-mounted.
- Conveyor moves continuously; the robot should intercept an object only after motion prediction and timing validation.
- Start with stationary objects, then introduce conveyor motion.
- Use Isaac depth as the initial source for simulated picking; compare against any RGB-derived depth method separately.
- Separate bins by object class. Keep the 60-class color/shape/apparent-size IDs stable.

## Implementation gates

- Gate A: audit labels and class order; train and evaluate 2D detector.
- Gate B: live RGB inference with frame timestamp and class/confidence overlay.
- Gate C: tracking with ID persistence, lost-track handling, and conveyor-motion estimation.
- Gate D: camera calibration, depth registration, coordinate transforms, and world-coordinate validation against known simulated points.
- Gate E: safe grasp planning in stationary scene.
- Gate F: moving-conveyor interception in simulation; validate missed picks, duplicate picks, and stop/recovery paths.

Do not send robot commands based only on 2D boxes. A detection is not a grasp pose. Validate coordinate transforms, depth, target age, reachability, collision clearance, and safety state before execution.
