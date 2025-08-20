#!/usr/bin/env python3
"""
main.py — Train and evaluate a YOLOv11 model to detect PCB defects

Dataset layout (Ultralytics format) is referenced by a data.yaml located at
`pcb-defect-dataset/data.yaml` with the following content:

path: ../pcb-defect-dataset
train: train
val: val
test: test

names:
  0: mouse_bite
  1: spur
  2: missing_hole
  3: short
  4: open_circuit
  5: spurious_copper

This script trains, validates, tests, and runs sample predictions.

Requirements (install once):
    pip install ultralytics pyyaml rich

Example usage:
    python main.py --epochs 100 --imgsz 640 --model yolo11n.pt --data pcb-defect-dataset/data.yaml --project runs_pcb --name y11_pcb --device 0

Tip: start with yolo11n.pt for speed; scale up to yolo11s.pt/yolo11m.pt for accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import yaml
from rich.console import Console
from rich.table import Table

from ultralytics import YOLO


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, validate, and test YOLOv11 on a PCB defect dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Paths / data
    parser.add_argument("--data", type=str, default="pcb-defect-dataset/data.yaml", help="Path to Ultralytics data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained YOLOv11 model checkpoint (.pt)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training/val")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer (auto, SGD, Adam, AdamW)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")

    # System
    parser.add_argument("--device", type=str, default="", help="CUDA device id like '0', or CPU if empty")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logging / outputs
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory for outputs")
    parser.add_argument("--name", type=str, default="y11_pcb", help="Run name (subfolder)")
    parser.add_argument("--save_json", action="store_true", help="Save metrics to JSON file")
    parser.add_argument("--export_onnx", action="store_true", help="Export best model to ONNX after training")
    parser.add_argument("--predict_samples", type=int, default=12, help="How many test images to run inference on and save")

    return parser.parse_args()


def read_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_images(data_yaml: Path, split: str) -> List[Path]:
    """Return a list of image paths for a given split using information in data.yaml.
    It tries common Ultralytics layouts: `<base>/<split>/images/*` or `<base>/<split>/*`.
    """
    cfg = read_yaml(data_yaml)
    base = Path(cfg.get("path", data_yaml.parent)).expanduser().resolve()
    split_rel = cfg.get(split)
    if split_rel is None:
        return []
    split_path = (base / split_rel).resolve()

    candidates = []
    # Prefer <split>/images
    images_dir = split_path / "images"
    if images_dir.is_dir():
        candidates.extend(sorted([p for p in images_dir.rglob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]))
    # Fallback: files directly under <split>
    if not candidates and split_path.is_dir():
        candidates.extend(sorted([p for p in split_path.rglob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]))
    return candidates


def pretty_print_metrics(title: str, metrics: Dict):
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)


def save_metrics_json(path: Path, **sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    console.print(f"[green]Saved metrics to[/green] {path}")


def train_and_evaluate(args: argparse.Namespace) -> None:
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    console.rule("[bold cyan]1) Load model")
    model = YOLO(args.model)  # e.g., 'yolo11n.pt' — COCO-pretrained weights

    console.rule("[bold cyan]2) Train")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    # results.save_dir holds the run folder; the model object now points to the last trained weights
    run_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    if best_pt.exists():
        model = YOLO(str(best_pt))  # reload best weights for eval/predict

    console.rule("[bold cyan]3) Validate (val split)")
    val_res = model.val(data=str(data_yaml), imgsz=args.imgsz, split="val")
    # The returned object exposes key metrics via .results_dict
    val_metrics = getattr(val_res, "results_dict", {})
    pretty_print_metrics("Validation Metrics (val)", val_metrics)

    console.rule("[bold cyan]4) Test (test split)")
    test_metrics = {}
    try:
        test_res = model.val(data=str(data_yaml), imgsz=args.imgsz, split="test")
        test_metrics = getattr(test_res, "results_dict", {})
        if test_metrics:
            pretty_print_metrics("Test Metrics (test)", test_metrics)
    except Exception as e:
        console.print(f"[yellow]Skipping test split evaluation:[/yellow] {e}")

    if args.save_json:
        save_metrics_json(
            run_dir / "metrics.json",
            validation=val_metrics,
            test=test_metrics,
            settings={
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "model": args.model,
                "seed": args.seed,
            },
        )

    console.rule("[bold cyan]5) Sample predictions (from test split)")
    try:
        test_images = resolve_split_images(data_yaml, "test")
        if not test_images:
            console.print("[yellow]No test images found to predict. Skipping predictions.[/yellow]")
        else:
            sample_imgs = test_images[: max(1, args.predict_samples)]
            pred_dir = run_dir / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            model.predict(
                source=[str(p) for p in sample_imgs],
                imgsz=args.imgsz,
                save=True,
                project=str(pred_dir),
                name="test_samples",
                exist_ok=True,
                verbose=False,
            )
            console.print(f"[green]Saved prediction images to[/green] {pred_dir / 'test_samples'}")
    except Exception as e:
        console.print(f"[yellow]Prediction step skipped:[/yellow] {e}")

    if args.export_onnx:
        console.rule("[bold cyan]6) Export to ONNX")
        try:
            export_path = model.export(format="onnx")
            console.print(f"[green]Exported ONNX model to[/green] {export_path}")
        except Exception as e:
            console.print(f"[red]ONNX export failed:[/red] {e}")

    console.rule("[bold green]Done")


if __name__ == "__main__":
    args = parse_args()
    # Make sure output root exists
    Path(args.project).mkdir(parents=True, exist_ok=True)
    # Respect env var for determinism if user sets it; ultralytics handles seeds internally
    os.environ.setdefault("ULTRALYTICS_VERBOSE", "True")
    train_and_evaluate(args)
