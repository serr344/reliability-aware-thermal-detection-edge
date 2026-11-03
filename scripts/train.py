"""Fine-tune a COCO-pretrained YOLO model on the person-only HIT-UAV dataset."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained weights to start from")
    parser.add_argument("--data", default="data/processed/hit-uav-person/data.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="smoke_test", help="Run name, results go to runs/detect/<name>")
    args = parser.parse_args()

    # Loads the architecture + COCO weights (downloaded automatically the first time)
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,          # data-loading processes; keep low on Windows
        seed=0,             # same seed -> comparable runs
        # Absolute path: Ultralytics nests a relative path under its own runs/detect folder
        project=str(Path("runs/detect").resolve()),
        name=args.name,
    )


# Required on Windows: DataLoader workers re-import this file,
# and without this guard each worker would start a new training.
if __name__ == "__main__":
    main()
