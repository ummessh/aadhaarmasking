import cv2
import numpy as np
import traceback
from typing import Dict, Tuple

from src.image_processing import (
    advanced_image_preprocessing_v2,
    detect_orientation_and_correct,
    create_tta_variants,
    enhanced_text_extraction,
)
from src.detector import EnhancedEnsembleDetector
from src.classifier import FalseNegativePreventionClassifier

def adjust_detections_for_variant(detections: list, variant_name: str, original_shape: tuple, variant_shape: tuple) -> list:
    if variant_name == "original" or not detections:
        return detections

    adjusted = []
    orig_h, orig_w = original_shape[:2]
    var_h, var_w = variant_shape[:2]

    if var_w == 0 or var_h == 0 or orig_w == 0 or orig_h == 0:
        return detections

    for detection in detections:
        try:
            adjusted_det = detection.copy()
            box = detection.get('box', [0, 0, 0, 0])
            if len(box) != 4: continue
            x1, y1, x2, y2 = box

            if "scale" in variant_name:
                scale_x, scale_y = orig_w / var_w, orig_h / var_h
                adjusted_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            elif "flip" in variant_name:
                adjusted_box = [orig_w - x2, y1, orig_w - x1, y2]
            else:
                adjusted_box = box

            adjusted_box = [
                max(0, min(orig_w, adjusted_box[0])),
                max(0, min(orig_h, adjusted_box[1])),
                max(0, min(orig_w, adjusted_box[2])),
                max(0, min(orig_h, adjusted_box[3])),
            ]

            if adjusted_box[0] < adjusted_box[2] and adjusted_box[1] < adjusted_box[3]:
                adjusted_det['box'] = adjusted_box
                adjusted.append(adjusted_det)
        except Exception as e:
            print(f"⚠️ Error adjusting detection for {variant_name}: {e}")
            adjusted.append(detection)
    return adjusted

def process_image(image: np.ndarray, ensemble_detector: EnhancedEnsembleDetector, classifier: FalseNegativePreventionClassifier) -> Tuple[Dict, Dict]:
    """
    Runs the complete classification pipeline on a single image.
    """
    try:
        print("🚀 Starting Aadhaar Classification Pipeline...")
        
        # Stage 1: Preprocessing
        processed_image = advanced_image_preprocessing_v2(image)
        corrected_image = detect_orientation_and_correct(processed_image)
        
        corrected_image = np.ascontiguousarray(corrected_image, dtype=np.uint8)

        # Stage 2: TTA
        tta_variants = create_tta_variants(corrected_image)

        # Stage 3: Detection
        all_detections = []
        for variant_image, variant_name in tta_variants:
            variant_detections = ensemble_detector.run_ensemble_detection(variant_image)
            if variant_detections and 'detections' in variant_detections:
                adjusted = adjust_detections_for_variant(
                    variant_detections['detections'], variant_name, image.shape, variant_image.shape
                )
                all_detections.extend(adjusted)

        # Stage 4: NMS
        consensus_detections = ensemble_detector.consensus_based_nms(all_detections)

        # Stage 5: OCR
        ocr_data = enhanced_text_extraction(corrected_image)

        # Stage 6: Classification
        classification_result = classifier.multi_stage_classification(
            consensus_detections, ocr_data, image
        )

        # Stage 7: Reporting
        detection_result = {
            'detections': consensus_detections,
            'total_raw_detections': len(all_detections),
            'ocr_confidence': ocr_data.get('confidence', 0.0),
        }
        
        print("✅ Pipeline completed successfully!")
        return detection_result, classification_result

    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        traceback.print_exc()
        return {'error': str(e)}, {'is_aadhaar': False, 'confidence': 0.0, 'evidence': [f"System error: {str(e)}"]}
