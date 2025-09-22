import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import io
import zipfile
import fitz  # PyMuPDF
import logging
import os
import tempfile

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
            "2. /label-image/": "Run labeling model on a single Aadhaar image",
            "3. /mask-image/": "Mask Aadhaar image using labeling results"
        }
    }


# -----------------------
# 1. CLASSIFICATION API
# -----------------------
@app.post("/classify-zip/")
def classify_zip(file: UploadFile = File(...)):
    """Classify images inside a ZIP file and return Aadhaar candidates."""
    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(status_code=400, detail="File must be a ZIP archive.")

    results = []
    zip_bytes = file.file.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        for filename in zf.namelist():
            image = None

            # Handle image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image = cv2.imdecode(np.frombuffer(zf.read(filename), np.uint8), cv2.IMREAD_COLOR)

            # Handle PDF files
            elif filename.lower().endswith('.pdf'):
                pdf_document = fitz.open(stream=zf.read(filename), filetype="pdf")
                for page_num in range(len(pdf_document)):
                    pix = pdf_document.load_page(page_num).get_pixmap(dpi=300)
                    image = cv2.imdecode(np.frombuffer(pix.tobytes("ppm"), np.uint8), cv2.IMREAD_COLOR)

            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                _, classification_result = process_image(
                    image_rgb,
                    ml_models["ensemble_detector"],
                    ml_models["classifier"]
                )
                results.append({
                    "file": filename,
                    "is_aadhaar": classification_result.get("is_aadhaar", False),
                    "confidence": classification_result.get("confidence", 0.0)
                })

    return JSONResponse(content={"candidates": results})


# -----------------------
# 2. LABELING API
# -----------------------
@app.post("/label-image/")
def label_image(file: UploadFile = File(...)):
    """Run ensemble detector (labeling model) on a single image."""
    image_bytes = file.file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Run detector
    detections = ml_models["ensemble_detector"].detect(image)
    return JSONResponse(content={"labels": detections})


# -----------------------
# 3. MASKING API
# -----------------------
@app.post("/mask-image/")
def mask_image(file: UploadFile = File(...)):
    """Mask Aadhaar image using the masking script."""
    temp_dir = tempfile.TemporaryDirectory()
    try:
        # Save uploaded image
        temp_path = os.path.join(temp_dir.name, "aadhaar.jpg")
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        # Run masking
        masked_path = mask_image_main(image_path=temp_path, verbose=False)
        if not masked_path or not os.path.exists(masked_path):
            raise HTTPException(status_code=500, detail="Masking failed.")

        return StreamingResponse(open(masked_path, "rb"), media_type="image/jpeg")

    finally:
        temp_dir.cleanup()


# -----------------------
# Run with Uvicorn
# -----------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Aadhaar Masking API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
