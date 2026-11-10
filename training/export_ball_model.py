"""Export the trained motion ball detector for the live pipeline.

  onnx    onnxruntime on CPU or CUDA (tracking/ball_model.py OnnxBackend); dynamic
          batch, so SEARCH can send every court tile at once
  engine  TensorRT FP16; must be built on the GPU that will run it (e.g. on the
          Kaggle T4), for batches up to --batch crops

Usage (from project root):
    python training/export_ball_model.py --weights runs/ball_motion/train/weights/best.pt
    python training/export_ball_model.py --weights best.pt --format engine
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracking.ball_model import CROP


def export(weights, fmt="onnx", batch=16):
    from ultralytics import YOLO
    model = YOLO(weights)
    if fmt == "onnx":
        out = model.export(format="onnx", imgsz=CROP, opset=17, dynamic=True, simplify=True)
    elif fmt == "engine":
        out = model.export(format="engine", imgsz=CROP, half=True, dynamic=True, batch=batch)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    print(f"Exported -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Export the motion ball detector")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    parser.add_argument("--batch", type=int, default=16, help="largest batch (engine)")
    args = parser.parse_args()
    export(args.weights, args.format, args.batch)


if __name__ == "__main__":
    main()
