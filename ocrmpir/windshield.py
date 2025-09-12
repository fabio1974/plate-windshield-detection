# windshield.py – detecção global (sem MIRA/Header)
from __future__ import annotations
from typing import Tuple, Optional
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

Rect = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

_WS = {
    "model_path": "runs/windshield/exp/weights/best.pt",
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    "device": None,          # "cuda", "cpu", "mps" ou None (auto)
    "pad_px": 2,
    "class_name": "windshield",
    "class_id": 0,
    # filtros opcionais:
    "min_area_frac": 0.01,   # descarta boxes < 1% da área da imagem (0 = desliga)
    "topk": 1,               # quantos boxes retornar (1 = melhor; pode aumentar se quiser lista)
}

_model = None  # lazy load

def set_windshield_params(**kwargs):
    _WS.update(kwargs)

def _clip_rect(x1, y1, x2, y2, w, h) -> Rect:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def _load_model():
    global _model
    if _model is not None:
        return _model
    if YOLO is None:
        raise RuntimeError("Pacote 'ultralytics' não encontrado. Instale com: pip install ultralytics")
    mp = _WS["model_path"]
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"Modelo não encontrado em: {mp}")
    _model = YOLO(mp)
    return _model

def _area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)

def detect_windshield_rect(img: np.ndarray) -> Optional[Rect]:
    """
    Detecta o melhor para-brisa na IMAGEM TODA.
    Retorna (x1,y1,x2,y2) absoluto ou None.
    """
    H, W = img.shape[:2]
    if H == 0 or W == 0:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = _load_model()
    results = model.predict(
        source=rgb,
        conf=_WS["conf"],
        iou=_WS["iou"],
        imgsz=_WS["imgsz"],
        device=_WS["device"],
        verbose=False
    )
    if not results:
        return None
    r = results[0]
    if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
        return None

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss  = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else None

    # filtra por classe, área mínima, ordena por confiança->área
    cand = []
    min_area = _WS["min_area_frac"] * (W * H)
    names = getattr(model.model, "names", None) or getattr(model, "names", None)

    for i in range(len(xyxy)):
        if clss is not None:
            if clss[i] != _WS["class_id"]:
                if names is None or str(names[clss[i]]).lower() != str(_WS["class_name"]).lower():
                    continue
        box = xyxy[i].astype(float)
        a = _area(box)
        if min_area > 0 and a < min_area:
            continue
        cand.append((float(confs[i]), a, box))

    if not cand:
        return None

    # ordena: maior confiança, depois maior área
    cand.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = cand[0][2]
    x1, y1, x2, y2 = best

    # padding e clipping
    x1 -= _WS["pad_px"]; y1 -= _WS["pad_px"]
    x2 += _WS["pad_px"]; y2 += _WS["pad_px"]
    return _clip_rect(x1, y1, x2, y2, W, H)

# compat: se seu código antigo ainda chama esta função, redireciona para global
def detect_windshield_rect_in_mira(img: np.ndarray, mira_rect: Rect) -> Optional[Rect]:
    """Mantido por compatibilidade. Agora ignora a MIRA e usa a imagem inteira."""
    return detect_windshield_rect(img)

def draw_rect(img: np.ndarray, rect: Rect, color=(255, 0, 255), thickness=2):
    if not rect:
        return
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
