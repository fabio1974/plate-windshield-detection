# train_windshield.py
import argparse
import os

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser("Treino YOLOv8 - Windshield")
    p.add_argument(
        "--data",
        default="datasets/windshield/data.yaml",
        help="caminho para o data.yaml exportado do Roboflow",
    )
    p.add_argument(
        "--model", default="yolov8n.pt", help="modelo base (yolov8n.pt | yolov8s.pt | ...)"
    )
    p.add_argument("--epochs", type=int, default=100, help="nº de épocas")
    p.add_argument("--imgsz", type=int, default=640, help="tamanho da imagem")
    p.add_argument("--batch", type=int, default=16, help="batch size")
    p.add_argument("--project", default="runs/windshield", help="pasta raiz de saída do YOLO")
    p.add_argument("--name", default="exp", help="subpasta do experimento")
    p.add_argument("--device", default="auto", help="gpu id, 'cpu' ou 'auto'")
    p.add_argument("--workers", type=int, default=8, help="nº de workers de dataloader")
    p.add_argument("--patience", type=int, default=25, help="early stopping patience (épocas)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    return p.parse_args()


def main():
    args = parse_args()

    # Detecta automaticamente o device
    try:
        import torch

        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        has_cuda = False

    if args.device.lower() == "auto":
        args.device = "0" if has_cuda else "cpu"

    os.environ.setdefault("WANDB_DISABLED", "true")  # silencia W&B

    print("🔹 Treinando com:")
    print(f"   data:       {args.data}")
    print(f"   model:      {args.model}")
    print(f"   epochs:     {args.epochs}")
    print(f"   imgsz:      {args.imgsz}")
    print(f"   batch:      {args.batch}")
    print(f"   device:     {args.device}")
    print(f"   out:        {args.project}/{args.name}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,  # agora é "0" (GPU) ou "cpu"
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        patience=args.patience,
        plots=True,
        pretrained=True,
    )

    print("\n✅ Treino finalizado!")
    print(f"📂 Melhor peso em: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
