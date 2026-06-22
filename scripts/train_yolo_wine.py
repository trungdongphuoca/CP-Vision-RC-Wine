import os
import sys
import shutil
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg

def create_data_yaml():
    """Create data.yaml configuration file for YOLOv8 training."""
    yaml_content = f"""path: {ROOT.as_posix()}/synthetic_dataset
train: images
val: images

nc: 4
names:
  0: brand
  1: vintage
  2: variety
  3: region
"""
    yaml_path = ROOT / "synthetic_dataset" / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[YOLO Train] Created data.yaml at {yaml_path}")
    return yaml_path

def main():
    print("=" * 65)
    print("  YOLOv8 Fine-Tuning for Wine Label Parsing")
    print("=" * 65)

    # 1. Check if synthetic dataset exists
    images_dir = ROOT / "synthetic_dataset" / "images"
    if not images_dir.exists() or len(os.listdir(images_dir)) == 0:
        print("[YOLO Train] Synthetic dataset not found. Generating...")
        from generate_synthetic_dataset import main as run_generator
        run_generator()

    # 2. Create data.yaml
    yaml_path = create_data_yaml()

    # 3. Load YOLO11 model
    from ultralytics import YOLO
    
    # We load the base nano model
    base_model_path = ROOT / "yolo11n.pt"
    if not base_model_path.exists():
        print("[YOLO Train] downloading yolo11n.pt...")
        model = YOLO("yolo11n.pt")  # downloads automatically
    else:
        model = YOLO(str(base_model_path))

    # 4. Train model
    # We train for 3 epochs as a demonstration / validation run.
    # In a full research setup, one should train for 50-100 epochs.
    print("[YOLO Train] Starting YOLO11 training...")
    results = model.train(
        data=str(yaml_path),
        epochs=3,
        imgsz=640,
        batch=8,
        workers=0,  # avoid multiprocessing issues on Windows
        project=str(ROOT / "runs"),
        name="yolo11_wine",
        verbose=False,
        device="cpu"
    )

    # 5. Save best weights
    best_weights = ROOT / "runs" / "yolo11_wine" / "weights" / "best.pt"
    target_path = ROOT / "models" / "yolo11_wine.pt"
    os.makedirs(ROOT / "models", exist_ok=True)

    if best_weights.exists():
        shutil.copy(best_weights, target_path)
        print(f"\n[YOLO Train] Success! Fine-tuned YOLO11 model saved to {target_path}")
    else:
        # Fallback to saving model state directly
        model.save(str(target_path))
        print(f"\n[YOLO Train] Model state saved to {target_path}")

    print("=" * 65)

if __name__ == "__main__":
    main()
