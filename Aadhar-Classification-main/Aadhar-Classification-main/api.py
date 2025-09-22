import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import io
import zipfile
import fitz  # PyMuPDF
import logging
import os
import tempfile

# This imports the main function from your masking script.
# Ensure 'prollyfinalam.py' is in the same directory.
from masking import main as mask_image_main

# Assuming your classifier is in a 'src' folder
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.detector import EnhancedEnsembleDetector
from src.classifier import FalseNegativePreventionClassifier
from src.pipeline import process_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to hold our models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML models during startup."""
    print("🚀 Loading models and initializing pipeline...")
    ml_models["ensemble_detector"] = EnhancedEnsembleDetector()
    ml_models["ensemble_detector"].load_ensemble_models()
    ml_models["classifier"] = FalseNegativePreventionClassifier()
    print("✅ Models loaded and pipeline ready.")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Aadhaar Masking API. Send a POST request to /process-and-mask-zip/ to process a ZIP file."}

# --- CHANGE 1: REMOVED 'async' FROM THE FUNCTION DEFINITION ---
@app.post("/process-and-mask-zip/")
def process_zip_and_mask(file: UploadFile = File(...)):
    """
    Accepts a ZIP file, finds the most likely Aadhaar card, masks it,
    and returns the masked image. This is a synchronous endpoint to handle blocking tasks.
    """
    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(status_code=400, detail="File provided is not a ZIP file.")

    temp_dir = tempfile.TemporaryDirectory()
    try:
        logger.info("Starting ZIP file processing.")
        # --- CHANGE 2: REMOVED 'await' AND MODIFIED THE READ METHOD ---
        zip_contents = file.file.read()
        
        zip_buffer = io.BytesIO(zip_contents)
        aadhaar_candidates = []

        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            for filename in zf.namelist():
                image_list = []
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_bytes = zf.read(filename)
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        image_list.append({"image": image, "source": filename})
                elif filename.lower().endswith('.pdf'):
                    pdf_bytes = zf.read(filename)
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page_num in range(len(pdf_document)):
                        page = pdf_document.load_page(page_num)
                        pix = page.get_pixmap(dpi=300)
                        image = cv2.imdecode(np.frombuffer(pix.tobytes("ppm"), np.uint8), cv2.IMREAD_COLOR)
                        if image is not None:
                            image_list.append({"image": image, "source": f"{filename} (page {page_num + 1})"})
                
                for item in image_list:
                    logger.info(f"Classifying image from: {item['source']}")
                    image_rgb = cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB)
                    _, classification_result = process_image(
                        image_rgb,
                        ml_models["ensemble_detector"],
                        ml_models["classifier"]
                    )
                    
                    if classification_result.get('is_aadhaar'):
                        logger.info(f"Aadhaar candidate found in {item['source']}")
                        aadhaar_candidates.append({
                            "source": item["source"],
                            "confidence": classification_result.get('confidence', 0.0),
                            "image": item["image"]
                        })

        if not aadhaar_candidates:
            raise HTTPException(status_code=404, detail="No Aadhaar card found in the ZIP file.")

        best_candidate = sorted(aadhaar_candidates, key=lambda x: x['confidence'], reverse=True)[0]
        logger.info(f"Best candidate is '{best_candidate['source']}'. Proceeding to mask.")

        # --- INTEGRATION POINT (LOGIC UNCHANGED) ---
        temp_image_path = os.path.join(temp_dir.name, "temp_aadhaar_image.jpg")
        cv2.imwrite(temp_image_path, best_candidate["image"])
        logger.info(f"Temporarily saved best candidate image to {temp_image_path}")
        
        masked_output_path = mask_image_main(image_path=temp_image_path, verbose=False)
        
        if not masked_output_path or not os.path.exists(masked_output_path):
            raise HTTPException(status_code=500, detail="Masking process failed to produce an output file.")
        
        logger.info(f"Masking successful. Masked file at: {masked_output_path}")

        with open(masked_output_path, "rb") as f:
            masked_image_bytes = f.read()
        
        return StreamingResponse(io.BytesIO(masked_image_bytes), media_type="image/jpeg")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        temp_dir.cleanup()
        logger.info("Cleaned up temporary files.")

if __name__ == "__main__":
    import uvicorn
    print("Starting Aadhaar Masking API server...")
    print("Visit http://127.0.0.1:8000/docs for the interactive API documentation.")
    uvicorn.run(app, host="127.0.0.1", port=8000)