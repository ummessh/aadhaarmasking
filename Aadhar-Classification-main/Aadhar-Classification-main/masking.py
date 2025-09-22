import cv2
import numpy as np
import re
import os
from typing import List, Tuple, Dict, Any, Optional
from paddleocr import PaddleOCR

# --- All helper functions remain unchanged ---

def format_ocr_result_to_json_structure(ocr_result: list, image_path: str) -> Optional[Dict[str, Any]]:
    """
    Converts the new, detailed dictionary output from PaddleOCR into the 
    legacy structured format that the rest of the script expects.
    """
    if not ocr_result or not isinstance(ocr_result[0], dict):
        print("Warning: OCR process returned an unexpected or empty format.")
        return None
    result_dict = ocr_result[0]
    rec_texts = result_dict.get('rec_texts', [])
    dt_polys = result_dict.get('dt_polys', [])
    if not rec_texts or not dt_polys:
        print("Warning: OCR did not detect both text and bounding boxes.")
        return None
    rec_polys_as_lists = [poly.tolist() for poly in dt_polys]
    return {
        'input_path': image_path,
        'rec_texts': rec_texts,
        'rec_polys': rec_polys_as_lists,
    }

def find_qr_code_bounding_boxes(image_path: str) -> List[Tuple[int, int, int, int]]:
    """Finds and returns the bounding box of the biggest QR code in an image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path} for QR code detection.")
        return []
    scale = 0.5
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    biggest_bbox = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        xmin, ymin, w, h = cv2.boundingRect(cnt)
        if w * h == 0: continue
        aspect_ratio = float(w) / h
        if (area > 100) and (0.85 <= aspect_ratio <= 1.15):
            if area > max_area:
                max_area = area
                original_xmin = int(xmin / scale)
                original_ymin = int(ymin / scale)
                original_xmax = int((xmin + w) / scale)
                original_ymax = int((ymin + h) / scale)
                biggest_bbox = (original_xmin, original_ymin, original_xmax, original_ymax)
    return [biggest_bbox] if biggest_bbox else []

def is_valid_aadhaar_number(text: str) -> bool:
    """Validate Aadhaar numbers."""
    cleaned = re.sub(r'\D', '', text)
    return len(cleaned) == 12 and not cleaned.startswith(('0', '1'))

def identify_sensitive_text(rec_texts: List[str]) -> List[Dict[str, Any]]:
    """
    Intelligently identifies sensitive text, with enhanced and more robust detection for DOB formats.
    """
    sensitive_matches = []
    processed_indices = set()
    month_regex = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    date_pattern_capture = rf"(\d{{1,2}}[\s/.-]+\d{{1,2}}[\s/.-]+\d{{4}}|\d{{4}}[\s/.-]+\d{{1,2}}[\s/.-]+\d{{1,2}}|\d{{1,2}}[\s/.-]+{month_regex}[\s/.-]+\d{{4}}|{month_regex}[\s/.-]+\d{{1,2}},?[\s/.-]+\d{{4}})"
    dob_label_pattern = r"(?:DOB|D\.O\.B|Date\s+of\s+Birth|Birth\s+Date)\s*:?\s*"
    context_keywords = ['dob', 'birth']
    for i, text in enumerate(rec_texts):
        if i in processed_indices or not text:
            continue
        text_cleaned = text.strip()
        full_pattern_labeled = dob_label_pattern + date_pattern_capture
        match = re.search(full_pattern_labeled, text_cleaned, re.IGNORECASE)
        if match:
            sensitive_matches.append({'indices': [i],'type': 'dob_labeled','original_text': text_cleaned,'mask_type': 'dob_partial','date_start_index': match.start(1)})
            processed_indices.add(i)
            continue
        match = re.search(date_pattern_capture, text_cleaned, re.IGNORECASE)
        if match:
            context_window_text = " ".join(filter(None, rec_texts[max(0, i - 2):min(len(rec_texts), i + 3)])).lower()
            if any(keyword in context_window_text for keyword in context_keywords):
                sensitive_matches.append({'indices': [i],'type': 'dob_generic','mask_type': 'full_date_mask'})
                processed_indices.add(i)
                continue
        aadhaar_pattern = r'\b(?:\d{4}\s?){2}\d{4}\b'
        if re.search(aadhaar_pattern, text_cleaned) and is_valid_aadhaar_number(text_cleaned):
            sensitive_matches.append({'indices': [i], 'type': 'aadhaar_single', 'mask_type': 'split_first_8_digits'})
            processed_indices.add(i)
    for i in range(len(rec_texts)):
        if i in processed_indices: continue
        current_digits = re.sub(r'\D', '', rec_texts[i] or '')
        if not current_digits: continue
        parts, indices = [current_digits], [i]
        for j in range(i + 1, min(i + 5, len(rec_texts))):
            if j in processed_indices: break
            next_digits = re.sub(r'\D', '', rec_texts[j] or '')
            if next_digits:
                parts.append(next_digits)
                indices.append(j)
                if len("".join(parts)) >= 12: break
        if is_valid_aadhaar_number("".join(parts)):
            sensitive_matches.append({'indices': indices, 'type': 'aadhaar_reconstructed', 'mask_type': 'split_first_8_digits'})
            processed_indices.update(indices)
    return sensitive_matches

def apply_all_masks_to_image(image_path: str, rec_texts: List[str], rec_boxes: List[List[int]], sensitive_matches: List[Dict[str, Any]], qr_code_boxes: List[Tuple[int, int, int, int]], output_path: str, mask_color: Tuple[int, int, int]) -> np.ndarray:
    """Applies masks for both sensitive text and QR codes onto a single image."""
    image = cv2.imread(image_path)
    if image is None: raise ValueError(f"Could not load image from {image_path} for masking.")
    masked_image = image.copy()
    for match in sensitive_matches:
        mask_type, indices = match['mask_type'], match['indices']
        if mask_type == 'split_first_8_digits':
            chars_to_mask = 8
            for index in indices:
                if chars_to_mask <= 0 or index >= len(rec_boxes): break
                digits = re.sub(r'\D', '', rec_texts[index] or '')
                if not digits: continue
                x1, y1, x2, y2 = rec_boxes[index]
                mask_chars = min(chars_to_mask, len(digits))
                mask_width = int((x2 - x1) * (mask_chars / len(digits)))
                if mask_width > 0: cv2.rectangle(masked_image, (x1, y1), (x1 + mask_width, y2), mask_color, -1)
                chars_to_mask -= mask_chars
        elif mask_type == 'dob_partial':
            index = indices[0]
            original_text = match.get('original_text', '')
            date_start_index = match.get('date_start_index')
            if index >= len(rec_boxes) or not original_text or date_start_index is None:
                continue
            x1, y1, x2, y2 = rec_boxes[index]
            box_width = x2 - x1
            mask_start_ratio = date_start_index / len(original_text)
            mask_start_x = x1 + int(box_width * mask_start_ratio)
            cv2.rectangle(masked_image, (mask_start_x, y1), (x2, y2), mask_color, -1)
        elif mask_type == 'full_date_mask':
            index = indices[0]
            if index >= len(rec_boxes): continue
            x1, y1, x2, y2 = rec_boxes[index]
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), mask_color, -1)
    for (xmin, ymin, xmax, ymax) in qr_code_boxes:
        cv2.rectangle(masked_image, (xmin, ymin), (xmax, ymax), mask_color, -1)
    cv2.imwrite(output_path, masked_image)
    return masked_image

def process_and_mask_document(ocr_data: Dict, output_path: str, mask_color: Tuple[int, int, int], verbose: bool) -> Dict[str, Any]:
    """Main function to process document, find all sensitive info, and mask it."""
    input_path = ocr_data['input_path']
    rec_texts = ocr_data['rec_texts']
    rec_polys = ocr_data['rec_polys']
    rec_boxes = [[min(p[0] for p in poly), min(p[1] for p in poly), max(p[0] for p in poly), max(p[1] for p in poly)] for poly in rec_polys]
    if verbose: print(f"--- Analyzing results for: {os.path.basename(input_path)} ---")
    sensitive_matches = identify_sensitive_text(rec_texts)
    qr_code_boxes = find_qr_code_bounding_boxes(input_path)
    if verbose:
        print(f"Found {len(sensitive_matches)} sensitive text matches.")
        print(f"Found {len(qr_code_boxes)} QR code(s).")
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_masked{ext}"
    if verbose: print(f"Applying masks and saving to {os.path.basename(output_path)}...")
    apply_all_masks_to_image(input_path, rec_texts, rec_boxes, sensitive_matches, qr_code_boxes, output_path, mask_color)
    return {
        'output_image_path': output_path,
        'total_text_matches': len(sensitive_matches),
        'total_qr_matches': len(qr_code_boxes),
    }

# --- MAIN CALLABLE FUNCTION (REFACTORED) ---

def main(image_path: str, output_path: Optional[str] = None, mask_color: Tuple[int, int, int] = (0, 0, 0), verbose: bool = True) -> Optional[str]:

    if not os.path.exists(image_path):
        print(f"Error: Input file not found at '{image_path}'")
        return None

    try:
        # Step 1: Initialize PaddleOCR engine
        if verbose: print("Initializing PaddleOCR engine...")
        ocr = PaddleOCR(
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False
        )

        # Step 2: Run OCR prediction
        if verbose: print(f"Running OCR on '{os.path.basename(image_path)}'...")
        result = ocr.predict(image_path)

        # Step 3: Format OCR result
        ocr_data = format_ocr_result_to_json_structure(result, image_path)
        if not ocr_data:
            print("Could not proceed with masking as no text was detected.")
            return None

        # Step 4: Process and mask the document
        final_results = process_and_mask_document(
            ocr_data=ocr_data,
            output_path=output_path,
            mask_color=mask_color,
            verbose=verbose
        )
        
        # Step 5: Report completion and return the output path
        if verbose:
            print("\n--- ✅ Processing Complete ---")
            print(f"Total sensitive text matches found: {final_results['total_text_matches']}")
            print(f"Total QR codes found: {final_results['total_qr_matches']}")
            print(f"\n➡️ Final masked image saved to: {final_results['output_image_path']}")
        
        return final_results['output_image_path']

    except Exception as e:
        print(f"\n--- ❌ An error occurred ---")
        print(f"Error: {e}")
        return None

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    """
    This block demonstrates how to call the main function from another script.
    It will only run when you execute this file directly.
    """
    # Define the path to your test image
    # IMPORTANT: Change this to a valid image path on your system
    test_image = r"C:\FeatSystems\Projects\Jaineel\Adhaarmask\Aadhar-Classification-main\test.jpg"
    
    print("--- Running Test ---")
    if os.path.exists(test_image):
        # Call the main function as you would from another file
        masked_file_path = main(image_path=test_image, verbose=True)
        
        if masked_file_path:
            print(f"\nTest successful. Masked file is at: {masked_file_path}")
        else:
            print("\nTest failed. Check errors above.")
    else:
        print(f"\nSkipping test: Please update the 'test_image' variable in the script with a valid path.")