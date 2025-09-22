import cv2
import numpy as np
import re
import easyocr
from typing import List, Dict, Tuple
from src.config import TTA_AUGMENTATIONS, ENHANCED_AADHAAR_PATTERNS, device

# Initialize OCR with multiple language support
# Citation for supported languages: https://www.jaided.ai/easyocr
try:
    # Using all documented Indian languages supported by EasyOCR for maximum coverage.
    languages = ['en', 'hi', 'as', 'bh', 'bho', 'bn', 'gom', 'kn', 'mah', 'mai', 'mr', 'ne', 'ta', 'te', 'ur', 'en']
    ocr_reader = easyocr.Reader(languages, gpu=device.type == 'cuda')
    print(f"✅ EasyOCR initialized with {len(languages)} languages based on official documentation.")
    print(f"   Languages: {', '.join(languages)}")
except Exception as e:
    print(f"⚠️ OCR initialization warning: {e}")
    ocr_reader = None

def create_tta_variants(image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """
    Create Test-Time Augmentation variants of input image - FIXED for any aspect ratio
    Citation: "TTA creates multiple augmented copies for ensemble predictions"
    """
    variants = []

    # Ensure image is in correct format (H, W, C)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Ensure image is properly formatted
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape[:2]

    # Original image - always include
    variants.append((image.copy(), "original"))

    # Scale variants - Research shows multi-scale improves detection
    # Fix: Handle any aspect ratio properly
    for scale in TTA_AUGMENTATIONS['scales']:
        try:
            if max(width, height) != scale:
                # Calculate scale factor maintaining aspect ratio
                scale_factor = scale / max(width, height)
                new_width = max(32, int(width * scale_factor))  # Ensure minimum size
                new_height = max(32, int(height * scale_factor))  # Ensure minimum size

                # Ensure dimensions are valid
                if new_width > 0 and new_height > 0:
                    scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                    # Ensure output is contiguous and correct format
                    scaled = np.ascontiguousarray(scaled, dtype=np.uint8)
                    variants.append((scaled, f"scale_{scale}"))
        except Exception as e:
            print(f"⚠️ Skipping scale {scale}: {e}")
            continue

    # Flip variants - Horizontal flip for orientation robustness
    try:
        if TTA_AUGMENTATIONS['flips'][1]:  # If horizontal flip enabled
            flipped = cv2.flip(image, 1)
            flipped = np.ascontiguousarray(flipped, dtype=np.uint8)
            variants.append((flipped, "flip_horizontal"))
    except Exception as e:
        print(f"⚠️ Skipping horizontal flip: {e}")

    # Rotation variants - Handle rotated Aadhaar cards with proper bounds
    for rotation in TTA_AUGMENTATIONS['rotations']:
        if rotation == 0:
            continue

        try:
            # Use simpler rotation for reliability
            if rotation == 90:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                # For other angles, use transformation matrix with bounds checking
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)

                # Calculate new dimensions properly
                cos_val = abs(rotation_matrix[0, 0])
                sin_val = abs(rotation_matrix[0, 1])
                new_width = max(32, int((height * sin_val) + (width * cos_val)))
                new_height = max(32, int((height * cos_val) + (width * sin_val)))

                # Adjust rotation matrix translation
                rotation_matrix[0, 2] += (new_width - width) / 2
                rotation_matrix[1, 2] += (new_height - height) / 2

                rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                       flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

            # Ensure output format
            rotated = np.ascontiguousarray(rotated, dtype=np.uint8)
            variants.append((rotated, f"rotation_{rotation}"))

        except Exception as e:
            print(f"⚠️ Skipping rotation {rotation}: {e}")
            continue

    # Brightness variants - Handle lighting variations
    for brightness in TTA_AUGMENTATIONS['brightness']:
        if brightness == 1.0:
            continue

        try:
            # More robust brightness adjustment
            bright_image = image.copy().astype(np.float32)
            bright_image = bright_image * brightness
            bright_image = np.clip(bright_image, 0, 255).astype(np.uint8)
            bright_image = np.ascontiguousarray(bright_image)
            variants.append((bright_image, f"brightness_{brightness}"))
        except Exception as e:
            print(f"⚠️ Skipping brightness {brightness}: {e}")
            continue

    print(f"🔄 Created {len(variants)} TTA variants for comprehensive detection")
    return variants

def advanced_image_preprocessing_v2(image: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing with research-based improvements
    Citation: "Image preprocessing techniques significantly improve detection accuracy"
    """
    # Ensure image is in correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB format is correct
        processed = image.copy()
    else:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Advanced denoising - Research shows this improves text detection
    processed = cv2.bilateralFilter(processed, 11, 80, 80)

    # Adaptive contrast enhancement
    lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Enhanced CLAHE with optimized parameters
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    processed = cv2.merge((l, a, b))
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)

    # Sharpening filter for better text detection
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    processed = cv2.filter2D(processed, -1, kernel)

    # Ensure values are in valid range
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    return processed

def detect_orientation_and_correct(image: np.ndarray) -> np.ndarray:
    """
    Advanced orientation detection and correction
    Citation: "Orientation correction significantly improves detection accuracy"
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Use edge detection for orientation
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

    if lines is not None:
        angles = []
        for line in lines[:20]:  # Use top 20 lines
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            if angle > 90:
                angle = angle - 180
            angles.append(angle)

        if angles:
            # Use median angle for robustness
            median_angle = np.median(angles)

            # Only correct if angle is significant
            if abs(median_angle) > 2:
                height, width = image.shape[:2]
                center = (width // 2, height // 2)

                # Create rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, -median_angle, 1.0)

                # Calculate new image dimensions
                cos_val = abs(rotation_matrix[0, 0])
                sin_val = abs(rotation_matrix[0, 1])
                new_width = int((height * sin_val) + (width * cos_val))
                new_height = int((height * cos_val) + (width * sin_val))

                # Adjust rotation matrix for new dimensions
                rotation_matrix[0, 2] += (new_width - width) / 2
                rotation_matrix[1, 2] += (new_height - height) / 2

                # Apply rotation
                corrected = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                return corrected

    return image

def enhanced_text_extraction(image: np.ndarray) -> Dict:
    """
    Advanced OCR with multiple extraction strategies
    """
    if not ocr_reader:
        return {'text': '', 'confidence': 0.0, 'patterns': {}}

    try:
        # Standard OCR
        results = ocr_reader.readtext(image)

        # Combine all text with confidence filtering
        all_text = []
        total_confidence = 0
        valid_results = 0

        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Filter low confidence
                all_text.append(text)
                total_confidence += confidence
                valid_results += 1

        combined_text = ' '.join(all_text)
        avg_confidence = total_confidence / max(valid_results, 1)

        # Pattern matching
        patterns_found = {}
        for pattern_type, patterns in ENHANCED_AADHAAR_PATTERNS.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, combined_text.lower())
                matches.extend(found)
            if matches:
                patterns_found[pattern_type] = matches

        return {
            'text': combined_text,
            'confidence': avg_confidence,
            'patterns': patterns_found,
            'raw_results': results
        }

    except Exception as e:
        print(f"⚠️ OCR extraction error: {e}")
        return {'text': '', 'confidence': 0.0, 'patterns': {}}
