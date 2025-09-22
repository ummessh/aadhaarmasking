import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from contextlib import asynccontextmanager
import io
import zipfile
import fitz  # PyMuPDF
import logging
import os
import tempfile
import json

# Import masking + models
from masking import main as mask_image_main
from src.detector import EnhancedEnsembleDetector
from src.classifier import FalseNegativePreventionClassifier
from src.pipeline import process_image

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary for models
ml_models = {}

# -----------------------
# Load Models on Startup
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading models...")
    ml_models["ensemble_detector"] = EnhancedEnsembleDetector()
    ml_models["ensemble_detector"].load_ensemble_models()
    ml_models["classifier"] = FalseNegativePreventionClassifier()
    logger.info("✅ Models loaded successfully.")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "Aadhaar Masking API",
        "endpoints": {
            "1. /classify-zip/": "Classify Aadhaar candidates inside a ZIP file",
            "2. /label-zip/": "Label Aadhaar/other docs in a ZIP and return annotated images",
            "3. /process-and-mask-zip/": "Find best Aadhaar candidate and return masked image"
        }
    }

# -----------------------
# Helper: classify images from ZIP
# -----------------------
def classify_zip_file(zip_bytes: bytes, ml_models: dict):
    results = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        for filename in zf.namelist():
            image_list = []

            # Handle image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image = cv2.imdecode(np.frombuffer(zf.read(filename), np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    image_list.append({"image": image, "source": filename})

            # Handle PDF files
            elif filename.lower().endswith('.pdf'):
                pdf_document = fitz.open(stream=zf.read(filename), filetype="pdf")
                for page_num in range(len(pdf_document)):
                    pix = pdf_document.load_page(page_num).get_pixmap(dpi=300)
                    image = cv2.imdecode(np.frombuffer(pix.tobytes("ppm"), np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        image_list.append({"image": image, "source": f"{filename} (page {page_num+1})"})

            # Run classification
            for item in image_list:
                image_rgb = cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB)
                _, classification_result = process_image(
                    image_rgb,
                    ml_models["ensemble_detector"],
                    ml_models["classifier"]
                )
                results.append({
                    "source": item["source"],
                    "is_aadhaar": classification_result.get("is_aadhaar", False),
                    "confidence": classification_result.get("confidence", 0.0),
                    "image": item["image"]  # keep for reuse in masking/labeling
                })

    return results

# -----------------------
# 1. CLASSIFY ZIP
# -----------------------
@app.post("/classify-zip/")
def classify_zip(file: UploadFile = File(...)):
    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(status_code=400, detail="File must be a ZIP archive.")
    try:
        results = classify_zip_file(file.file.read(), ml_models)
        return {"candidates": [
            {"source": r["source"], "is_aadhaar": r["is_aadhaar"], "confidence": r["confidence"]}
            for r in results
        ]}
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# -----------------------
# 2. LABEL ZIP
# -----------------------
@app.post("/label-zip/")
def label_zip(file: UploadFile = File(...)):
    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(status_code=400, detail="File must be a ZIP archive.")
    
    temp_dir = tempfile.TemporaryDirectory()
    output_dir = tempfile.mkdtemp()
    try:
        results = classify_zip_file(file.file.read(), ml_models)
        manifest = []

        for r in results:
            img = r["image"].copy()
            label = "AADHAAR" if r["is_aadhaar"] else "OTHER"

            # Draw label on image
            cv2.putText(img, f"{label} ({r['confidence']:.2f})",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            out_path = os.path.join(output_dir, f"labeled_{os.path.basename(r['source'])}.jpg")
            cv2.imwrite(out_path, img)

            manifest.append({"source": r["source"], "label": label, "confidence": r["confidence"]})

        # Create zip of outputs
        out_zip = os.path.join(temp_dir.name, "labeled_results.zip")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in os.listdir(output_dir):
                zf.write(os.path.join(output_dir, f), f)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        return FileResponse(out_zip, media_type="application/zip", filename="labeled_results.zip")

    finally:
        temp_dir.cleanup()
        logger.info("Cleaned up temporary files.")

# -----------------------
# 3. PROCESS + MASK ZIP
# -----------------------
@app.post("/process-and-mask-zip/")
def process_and_mask_zip(file: UploadFile = File(...)):
    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(status_code=400, detail="File must be a ZIP archive.")

    temp_dir = tempfile.TemporaryDirectory()
    try:
        results = classify_zip_file(file.file.read(), ml_models)
        aadhaar_candidates = [r for r in results if r["is_aadhaar"]]

        if not aadhaar_candidates:
            raise HTTPException(status_code=404, detail="No Aadhaar card found in the ZIP file.")

        best = sorted(aadhaar_candidates, key=lambda x: x["confidence"], reverse=True)[0]
        logger.info(f"Best Aadhaar candidate: {best['source']}")

        # Save best candidate to disk
        temp_image_path = os.path.join(temp_dir.name, "aadhaar.jpg")
        cv2.imwrite(temp_image_path, best["image"])

        # Run masking
        masked_output_path = mask_image_main(image_path=temp_image_path, verbose=False)
        if not masked_output_path or not os.path.exists(masked_output_path):
            raise HTTPException(status_code=500, detail="Masking failed.")

        return StreamingResponse(open(masked_output_path, "rb"), media_type="image/jpeg")

    finally:
        temp_dir.cleanup()

# -----------------------
# Run with Uvicorn
# -----------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Aadhaar Masking API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
