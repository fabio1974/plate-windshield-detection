import glob
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
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


file_dependency = File(...)


# New endpoint: accept image as bytes (multipart/form-data)
@app.post("/detect", response_model=PlateResponse)
async def detect_plate_bytes(file: UploadFile = file_dependency):
    try:
        contents = await file.read()
        img_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Could not decode image")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    # Optimized: no disk I/O for /detect
    plate, conf = pr.detect_and_read_plate_array(img_array)
    rect = detect_windshield_rect(img_array)
    return PlateResponse(
        plate=plate,
        confidence=conf,
        windshield=[int(x) for x in rect] if rect else None,
    )


# New endpoint: receive image as bytes and return annotated image as bytes
@app.post("/detect_image")
async def annotate(file: UploadFile = file_dependency):
    try:
        contents = await file.read()
        img_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    # Process and annotate
    old_flag = pr.SAVE_ANNOTATED
    pr.SAVE_ANNOTATED = False
    try:
        _plate, _conf, rect, annotated_path = process_image_array(img_array)
        if annotated_path:
            annotated = cv2.imread(annotated_path)
            if annotated is None:
                annotated = img_array.copy()
        else:
            annotated = img_array.copy()
        # Guarantee same dimensions as original
        if annotated.shape[:2] != img_array.shape[:2]:
            annotated = cv2.resize(
                annotated, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        if rect:
            draw_rect(annotated, rect, color=(255, 0, 255), thickness=2)
    finally:
        pr.SAVE_ANNOTATED = old_flag
    _, img_encoded = cv2.imencode(".png", annotated)
    return Response(content=img_encoded.tobytes(), media_type="image/png")


# Batch annotation endpoint for testing
@app.get("/detect_batch")
def detect_batch():
    folder = "/mpir/origem"
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files = []
    for ext in exts:
        files.extend(glob.glob(str(Path(folder) / f"*{ext}")))
    log = []
    for img_path in sorted(files):
        try:
            img = cv2.imread(img_path)
            plate, conf, rect, annotated_path = process_image_array(img)
            if annotated_path:
                annotated = cv2.imread(annotated_path)
                if annotated is None:
                    annotated = img.copy()
            else:
                annotated = img.copy()
            # Guarantee same dimensions as original
            if annotated.shape[:2] != img.shape[:2]:
                annotated = cv2.resize(
                    annotated, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            if rect:
                draw_rect(annotated, rect, color=(255, 0, 255), thickness=2)
            annotated_name = str(Path(img_path).with_suffix("").name) + "_annotated.jpg"
            annotated_path_final = str(Path(img_path).parent / annotated_name)
            cv2.imwrite(annotated_path_final, annotated)
            print(f"Processed {Path(img_path).name} -> {annotated_name}")
            log.append(
                {
                    "file": Path(img_path).name,
                    "plate": plate,
                    "confidence": conf,
                    "windshield": [int(x) for x in rect] if rect else None,
                }
            )
        except Exception as e:
            print(f"Error processing {Path(img_path).name}: {e}")
            log.append({"file": Path(img_path).name, "error": str(e)})
    return JSONResponse(content=log)


def process_image_array(img: "np.ndarray"):
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


def process_image(path: str, draw_ws: bool = True, save_annotated: bool = True):
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
