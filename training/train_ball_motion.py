"""Train the motion ball detector on the dataset of build_ball_dataset.py.

Runs ultralytics with the augmentations that make no sense for these images turned
off: the channels are not colours (no HSV jitter), and mosaic would halve the size
of a ball that is already ~15 px. The dataset builder applies its own augmentation
(gain on the differences, gamma on the grey image) when it cuts the crops.

Needs ultralytics and a GPU (see notebooks/ball_motion_training_kaggle.ipynb).

Usage (from project root):
    python training/train_ball_motion.py --data datasets/ball_motion/data.yaml
    python training/train_ball_motion.py --model yolov8s.pt --epochs 150
"""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracking.ball_model import CROP

TRAIN_ARGS = {
    "imgsz": CROP,
    "epochs": 100,
    "patience": 20,
    "batch": 64,
    # The RAM cache crashes ultralytics' trainer on Windows (native, no traceback).
    "cache": "ram" if sys.platform != "win32" else False,
    "single_cls": True,
    # Not colours: no HSV jitter.
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    # The ball must keep its size and the channels their meaning.
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "cutmix": 0.0,
    "degrees": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "erasing": 0.0,
    # Safe ones.
    "fliplr": 0.5, "scale": 0.2, "translate": 0.1,
}


def check_albumentations():
    """Ultralytics applies albumentations (grey, CLAHE, blur...) when installed."""
    if importlib.util.find_spec("albumentations") is not None:
        print("WARNING: albumentations is installed; ultralytics would mix the channels "
              "(ToGray, CLAHE). Run `pip uninstall -y albumentations` first.")
        return False
    return True


def train(data, model="yolo11n.pt", project="runs/ball_motion", name="train", **overrides):
    from ultralytics import YOLO
    args = dict(TRAIN_ARGS, **overrides)
    yolo = YOLO(model)
    yolo.train(data=data, project=project, name=name, exist_ok=True, **args)
    best = Path(yolo.trainer.best)
    print(f"Best weights -> {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Train the motion ball detector")
    parser.add_argument("--data", default="datasets/ball_motion/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="starting weights: yolo11n.pt, yolov8n.pt, yolov8s.pt...")
    parser.add_argument("--epochs", type=int, default=TRAIN_ARGS["epochs"])
    parser.add_argument("--batch", type=int, default=TRAIN_ARGS["batch"])
    parser.add_argument("--device", default=None, help="e.g. 0, or 0,1 for two GPUs")
    parser.add_argument("--project", default="runs/ball_motion")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()
    if not check_albumentations():
        raise SystemExit(1)
    extra = {"device": args.device} if args.device is not None else {}
    train(args.data, args.model, args.project, args.name, epochs=args.epochs,
          batch=args.batch, **extra)


if __name__ == "__main__":
    main()
