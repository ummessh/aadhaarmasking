# Enhanced Aadhaar Card Classification

This project is a robust system for detecting and classifying Aadhaar cards from images. It is a refactoring of a Jupyter Notebook into a structured Python project, optimized for performance and maintainability. The system leverages an ensemble of YOLO models, Test-Time Augmentation (TTA), and advanced classification logic to achieve high accuracy.

## Features

- **Multi-Model Ensemble**: Utilizes YOLOv8-nano, YOLOv11-nano, and YOLOv8-small for improved detection accuracy and robustness.
- **Test-Time Augmentation (TTA)**: Employs various image augmentation techniques at inference time (scaling, flipping, rotation, brightness adjustments) to enhance detection performance.
- **Advanced Image Preprocessing**: Includes orientation correction, denoising, and adaptive contrast enhancement.
- **OCR Integration**: Uses EasyOCR for text extraction and pattern matching to validate Aadhaar card fields.
- **Multi-Stage Classification**: A sophisticated pipeline that combines evidence from object detection and OCR to make a final classification decision.
- **False Negative Prevention**: Implements checks to minimize the chances of missing a genuine Aadhaar card.
- **GPU Support**: Automatically utilizes a CUDA-enabled GPU if available, falling back to CPU otherwise.

## Multi-Language Support

To ensure high accuracy across India, this project uses an OCR configuration that supports all officially recognized Indian languages available in the EasyOCR library.

-   **Source of Language List**: The supported languages are based on the [official EasyOCR documentation](https://www.jaided.ai/easyocr).
-   **Supported Indian Languages**: The engine is initialized with the following language codes: `en`, `hi`, `as`, `bh`, `bho`, `bn`, `gom`, `kn`, `mah`, `mai`, `mr`, `ne`, `ta`, `te`, `ur`.
-   **Keyword Recognition**: The system includes regular expression patterns to detect keywords like "Government of India" and gender indicators in multiple scripts, including Latin, Devanagari, Tamil, Telugu, Kannada, Bengali, Malayalam, and Urdu.

## Project Structure

```
Aadhar-Classification/
├── src/
│   ├── config.py               # Configuration for models, thresholds, and augmentations
│   ├── image_processing.py     # TTA, preprocessing, and OCR functions
│   ├── detector.py             # EnhancedEnsembleDetector class for model loading and detection
│   ├── classifier.py           # FalseNegativePreventionClassifier for classification logic
│   ├── pipeline.py             # Core processing pipeline
│   └── visualize.py            # Functions for visualizing results (used by CLI)
├── api.py                      # FastAPI server entry point
├── main.py                     # Deprecated CLI entry point
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- `pip` for package management
- For GPU support:
    - An NVIDIA GPU
    - CUDA Toolkit installed
    - cuDNN installed

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Aadhar-Classification
    ```

2.  **Create and activate a virtual environment:**

    A virtual environment is a self-contained directory that holds a specific Python interpreter and its own set of installed packages. Using one is highly recommended to avoid conflicts with other projects or your system-wide Python installation.

    ```bash
    python -m venv venv
    ```
    - **On Windows:**
      ```bash
      venv\Scripts\activate
      ```
    - **On macOS/Linux:**
      ```bash
      source venv/bin/activate
      ```
    You will know the environment is active when you see `(venv)` at the beginning of your command prompt.

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This may take some time as it will download PyTorch, CUDA-related libraries (if applicable), and other packages. The models will be downloaded to a `models/` directory upon first run.*

### Project Environment

This project includes a `.gitignore` file, which is configured to exclude the virtual environment (`venv/`), downloaded models (`models/`), and other unnecessary files from version control. This keeps the repository clean and focused on the source code.

### Hugging Face Token (Recommended)

To avoid potential download rate limits when fetching the pre-trained models from Hugging Face Hub, it is highly recommended to use an access token.

1.  **Create a `.env` file** in the root of the project directory. You can do this by copying the example file:
    ```bash
    # On Windows (Command Prompt)
    copy .env.example .env

    # On macOS/Linux
    cp .env.example .env
    ```
2.  **Generate a token** from your Hugging Face account settings: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3.  **Add the token** to your new `.env` file:
    ```
    HUGGING_FACE_HUB_TOKEN="your_token_here"
    ```
The application will automatically load this token when it runs.

## Running the API Server

This project is served as a web API using FastAPI.

1.  **Start the server:**
    Make sure your virtual environment is activated and you are in the root directory of the project. Then, run the following command:
    ```bash
    python -m uvicorn api:app --reload
    ```
    Using `python -m uvicorn` is a more reliable method that avoids potential `PATH` issues with your Python installation.

    The server will start on `http://127.0.0.1:8000`. The `--reload` flag enables auto-reloading when you make changes to the code.

2.  **Access the Interactive Documentation:**
    Once the server is running, you can access the interactive API documentation (powered by Swagger UI) by navigating to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your web browser.

## API Usage

To classify an image, send a `POST` request to the `/classify/` endpoint with the image file.

### Example using `curl`

```bash
curl -X POST -F "file=@/path/to/your/aadhar_card.jpg" http://127.0.0.1:8000/classify/
```

### Example using Python `requests`

Here is a simple Python script to send an image to the API and print the result:

```python
import requests
import json

# API endpoint URL
url = "http://127.0.0.1:8000/classify/"

# Path to the image file
image_path = "path/to/your/aadhar_card.jpg"

try:
    with open(image_path, "rb") as image_file:
        # Prepare the file for the POST request
        files = {"file": (image_path, image_file, "image/jpeg")}
        
        # Send the request
        response = requests.post(url, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Print the JSON response
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

except FileNotFoundError:
    print(f"Error: The file was not found at {image_path}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

### JSON Response Structure

The API will return a JSON object with the classification and detection details.

```json
{
  "filename": "your_image.jpg",
  "classification": {
    "is_aadhaar": true,
    "confidence": 0.95,
    "evidence": [
      "YOLO detected 5 field(s)",
      "OCR found 1 aadhaar_number pattern(s)"
    ],
    "detected_fields": ["AADHAR_NUMBER", "NAME", "DATE_OF_BIRTH"]
  },
  "detection_details": {
    "detections": [
      {
        "model": "ensemble_consensus",
        "box": [100, 200, 300, 250],
        "confidence": 0.98,
        "label": "AADHAR_NUMBER"
      }
    ],
    "total_raw_detections": 50,
    "ocr_confidence": 0.88
  }
}
```
