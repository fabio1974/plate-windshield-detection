import os

os.environ["GLOG_MINLOGLEVEL"] = "3"
os.environ["FLAGS_LOGTOSTDERR"] = "1"

import logging
import re
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

# =================== FLAGS (same as your original file) ===================
DEBUG = False  # True = saves crop variations
SAVE_FAILED = True  # saves crops when it fails / low confidence
LOW_CONF = 0.60  # low confidence threshold for "failure"
SAVE_CROPS = False  # False = never save crops
SAVE_ANNOTATED = False  # True = save *_annotated.png
# ================================================================================


# ---------- YOLO Weights (license plate) ----------
DETECTOR_WEIGHTS = "license_plate_detector.pt"
CONF_THRES = 0.25
IOU_THRES = 0.50
PAD = 6

# ---------- OCR ----------
ocr = PaddleOCR(lang="latin", det=False, rec=True, cls=False)


logging.getLogger("ppocr").setLevel(logging.ERROR)


# ---------- Brazilian license plate patterns ----------
PLATE_OLD = re.compile(r"\b[A-Z]{3}\d{4}\b")  # ABC1234
PLATE_MERC = re.compile(r"\b[A-Z]{3}\d[A-Z]\d{2}\b")  # ABC1D23


# ---------------- utils OCR -----------------
def normalize(text: str) -> str:
    t = text.strip().upper().replace(" ", "").replace("-", "")
    swaps = {"0": "O", "1": "I", "5": "S", "8": "B", "Q": "O"}
    return "".join(swaps.get(c, c) for c in t)


def strict_correction(t: str) -> str | None:
    t = normalize(t)
    if not re.fullmatch(r"[A-Z0-9]{7}", t):
        return None
    if PLATE_OLD.fullmatch(t) or PLATE_MERC.fullmatch(t):
        return t
    chars = list(t)
    for i in range(len(chars)):
        if i in (0, 1, 2) and chars[i].isdigit():
            chars[i] = {"0": "O", "1": "I", "5": "S", "8": "B"}.get(chars[i], chars[i])
        if i in (3, 4, 5, 6) and chars[i].isalpha():
            chars[i] = {"O": "0", "Q": "0", "D": "0", "B": "8", "I": "1", "S": "5"}.get(
                chars[i], chars[i]
            )
    t2 = "".join(chars)
    if PLATE_OLD.fullmatch(t2) or PLATE_MERC.fullmatch(t2):
        return t2
    return None


def _maybe_save(tag: str, img: np.ndarray, base_dir: Path | None, idx: int, force: bool = False):
    if not SAVE_CROPS or base_dir is None:
        return
    if DEBUG or force:
        base_dir.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(base_dir / f"crop_{idx:02d}_{tag}.png"), img)


def _best_plate_from_crop(
    crop: np.ndarray, save_dir: Path | None = None, idx: int = 0
) -> tuple[str | None, float]:
    def run_once(mat):
        recs = ocr.ocr(mat, cls=False) or []
        best, best_conf = None, 0.0
        for line in recs:
            if not line or not isinstance(line, (list, tuple)):
                continue
            for item in line:
                if not item:
                    continue
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                    txt, conf = item
                elif (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                ):
                    try:
                        _, (txt, conf) = item
                    except Exception:
                        continue
                else:
                    continue
                cand = strict_correction(txt) or normalize(txt)
                if (PLATE_OLD.fullmatch(cand) or PLATE_MERC.fullmatch(cand)) and float(
                    conf
                ) > best_conf:
                    best, best_conf = cand, float(conf)
        return best, best_conf

    crop0 = crop.copy()
    crop2 = cv2.resize(crop0, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    crop3 = cv2.resize(crop0, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    def unsharp(img, k=0.8):
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        return cv2.addWeighted(img, 1 + k, blur, -k, 0)

    def binarize(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    def adapt(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.bilateralFilter(g, 9, 75, 75)
        th = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    variants = [
        ("raw", crop0),
        ("x2", crop2),
        ("x3", crop3),
        ("sharp", unsharp(crop2)),
        ("otsu", binarize(crop2)),
        ("adapt", adapt(crop2)),
    ]

    best, best_conf = None, 0.0
    for tag, imgv in variants:
        if DEBUG:
            _maybe_save(tag, imgv, save_dir, idx)
        p, c = run_once(imgv)
        if p and c > best_conf:
            best, best_conf = p, c

    if (not best or best_conf < LOW_CONF) and SAVE_FAILED:
        for tag, imgv in variants:
            _maybe_save(f"FAIL_{tag}", imgv, save_dir, idx, force=True)

    return best, best_conf


def detect_and_read_plate(image_path: str) -> tuple[str | None, float, str | None]:
    """
    Reads the image **without any mask**, detects the plate (YOLO),
    runs OCR on the crop from the ORIGINAL and returns (plate, confidence, annotated_path_or_None).
    Respects the FLAGS declared above.
    """
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    img_for_detect = img_orig  # sem máscara

    # 1) YOLO (placa) na imagem integral
    model = YOLO(DETECTOR_WEIGHTS)
    res = model.predict(img_for_detect, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    detect_names_dynamic = {n.lower() for n in res.names.values() if "plate" in n.lower()}
    accepted_names = detect_names_dynamic or {"license-plate", "plate", "placa"}

    boxes = []
    boxes_obj = getattr(res, "boxes", None)
    if boxes_obj is not None and len(boxes_obj) > 0:
        for b in boxes_obj:
            cls_id = int(b.cls[0].item())
            name = res.names.get(cls_id, str(cls_id)).lower()
            if (name in accepted_names) or ("plate" in name):
                boxes.append(b)
        if not boxes:
            boxes = [max(boxes_obj, key=lambda z: float(z.conf[0]))]

    annotated = img_orig.copy() if SAVE_ANNOTATED else None
    best_global, best_conf = None, 0.0

    debug_dir = None
    if (DEBUG or SAVE_FAILED) and SAVE_CROPS:
        base = Path(image_path).with_suffix("")
        debug_dir = Path(str(base) + "_crops")

    # 2) OCR nos bboxes
    h, w = img_for_detect.shape[:2]
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
        x2, y2 = min(w, x2 + PAD), min(h, y2 + PAD)

        crop = img_orig[y1:y2, x1:x2]  # OCR no original
        plate, conf = _best_plate_from_crop(crop, save_dir=debug_dir, idx=i)

        if plate and conf > best_conf:
            best_global, best_conf = plate, conf

        if SAVE_ANNOTATED and annotated is not None:
            color = (0, 255, 0) if plate else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{plate or '??'} {conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    out_path = None
    if SAVE_ANNOTATED and annotated is not None:
        out_path = str(Path(image_path).with_suffix("")) + "_annotated.png"
        cv2.imwrite(out_path, annotated)

    return best_global, best_conf, out_path
