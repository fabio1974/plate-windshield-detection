import base64

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import ocrmpir.plate_reader as pr
from ocrmpir.windshield import (
    detect_windshield_rect,
    draw_rect,
    set_windshield_params,
)

app = FastAPI()


class PlateRequest(BaseModel):
    image_base64: str


class PlateResponse(BaseModel):
    plate: str | None
    confidence: float | None
    windshield: list | None
    annotated_path: str | None


def process_image_array(img: "np.ndarray"):
    """
    Detects the license plate and windshield from an image array.
    Returns: plate, confidence, windshield_rect (or None), annotated_path (or None)
    """
    old_flag = pr.SAVE_ANNOTATED
    pr.SAVE_ANNOTATED = True
    annotated_path = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            plate, conf, out = pr.detect_and_read_plate(tmp.name)
            annotated_path = out
            # Draw windshield if found
            rect = detect_windshield_rect(img)
            if rect and annotated_path:
                annotated = cv2.imread(annotated_path)
                if annotated is not None:
                    draw_rect(annotated, rect, color=(255, 0, 255), thickness=2)
                    cv2.imwrite(annotated_path, annotated)
            else:
                rect = detect_windshield_rect(img)
    finally:
        pr.SAVE_ANNOTATED = old_flag
    return plate, conf, rect, annotated_path


@app.post("/detect", response_model=PlateResponse)
def detect_plate(req: PlateRequest):
    try:
        img_bytes = base64.b64decode(req.image_base64)
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    plate, conf, rect, annotated_path = process_image_array(img_array)
    return PlateResponse(
        plate=plate,
        confidence=conf,
        windshield=[int(x) for x in rect] if rect else None,
        annotated_path=annotated_path,
    )


def process_image(path: str, draw_ws: bool = True, save_annotated: bool = True):
    """
    1) Detects the license plate normally (without masking header/target).
    2) Detects the windshield in the entire image and draws the rectangle.
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
    # Windshield detection parameters
    set_windshield_params(
        model_path="runs/windshield/exp/weights/best.pt",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device=None,  # or "cuda"
        pad_px=2,
        min_area_frac=0.01,
    )

    import uvicorn

    print("Starting HTTP service on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
