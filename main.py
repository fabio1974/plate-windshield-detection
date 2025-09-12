import glob
import sys
from pathlib import Path

import cv2

import ocrmpir.plate_reader as pr

# pacote
from ocrmpir.windshield import (
    detect_windshield_rect,
    draw_rect,
    set_windshield_params,
)


def process_image(path: str, draw_ws: bool = True, save_annotated: bool = True):
    """
    1) Detecta a placa normalmente (sem mascarar header/mira).
    2) Detecta o para-brisa na imagem inteira e desenha o retângulo.
    """
    old_flag = pr.SAVE_ANNOTATED
    pr.SAVE_ANNOTATED = save_annotated
    try:
        plate, conf, out = pr.detect_and_read_plate(path)
    finally:
        pr.SAVE_ANNOTATED = old_flag

    if not (draw_ws and save_annotated):
        return plate, conf, out

    img = cv2.imread(path)
    if img is None:
        return plate, conf, out

    rect = detect_windshield_rect(img)

    if out and rect:
        annotated = cv2.imread(out)
        if annotated is not None:
            draw_rect(annotated, rect, color=(255, 0, 255), thickness=2)
            cv2.imwrite(out, annotated)

    return plate, conf, out


if __name__ == "__main__":
    # Parâmetros de detecção de para-brisa
    set_windshield_params(
        model_path="runs/windshield/exp/weights/best.pt",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device=None,  # ou "cuda"
        pad_px=2,
        min_area_frac=0.01,
    )

    folder = "/mpir/origem"
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith((".jpg", ".jpeg", ".png")):
        imgs = [sys.argv[1]]
    else:
        imgs = sorted(
            glob.glob(str(Path(folder) / "*.jpg")) + glob.glob(str(Path(folder) / "*.JPG"))
        )

    if not imgs:
        print(f"Nenhuma imagem encontrada em {folder}")
        sys.exit(0)

    print(f"Processando {len(imgs)} imagens em {folder}...\n")
    for p in imgs:
        try:
            plate, conf, out = process_image(p, draw_ws=True, save_annotated=True)
            print(
                f"{Path(p).name} -> Placa: {plate} | Confiança: {conf:.2f} | Preview: {out or 'não salvo'}"
            )
        except Exception as e:
            print(f"{Path(p).name} -> ERRO: {e}")
