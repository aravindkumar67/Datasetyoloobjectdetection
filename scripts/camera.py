from __future__ import annotations
import argparse
import cv2
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser(description="Live webcam detection. Press q to quit.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0", help="Camera index such as 0, or a supported RTSP/HTTP stream URL")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--window", default="YOLO26 Live Detection")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera/stream source {args.source}. On a remote VM, a local laptop webcam may not be forwarded into the VM.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed; ending stream.")
                break
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            annotated = result.plot()
            cv2.imshow(args.window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__": raise SystemExit(main())
