import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from src.config import CLASSIFICATION_THRESHOLDS

def visualize_enhanced_results(image: np.ndarray, detection_result: Dict,
                             classification_result: Dict, filename: str):
    """
    Enhanced visualization with comprehensive results display
    """
    fig = plt.figure(figsize=(20, 16))

    # Main result display
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title(f'Original Image: {filename}', fontsize=14, weight='bold')
    ax1.axis('off')

    # Detection visualization
    ax2 = plt.subplot(2, 3, 2)
    vis_image = visualize_detections_enhanced(image, detection_result['detections'])
    ax2.imshow(vis_image)
    detection_count = len(detection_result.get('detections', []))
    ax2.set_title(f'Enhanced Detections ({detection_count} found)', fontsize=14, weight='bold')
    ax2.axis('off')

    # Results panel
    ax3 = plt.subplot(2, 3, (3, 6))
    ax3.axis('off')

    # Main classification result
    is_aadhaar = classification_result['is_aadhaar']
    confidence = classification_result['confidence']

    result_color = 'darkgreen' if is_aadhaar else 'darkred'
    result_text = '🎉 AADHAAR CARD DETECTED!' if is_aadhaar else '❌ NOT AN AADHAAR CARD'

    y_pos = 0.95
    ax3.text(0.05, y_pos, result_text, fontsize=20, weight='bold', color=result_color, transform=ax3.transAxes)
    y_pos -= 0.08

    ax3.text(0.05, y_pos, f'Final Confidence: {confidence:.1%}', fontsize=16, transform=ax3.transAxes)
    y_pos -= 0.05
    ax3.text(0.05, y_pos, f'Classification Threshold: {CLASSIFICATION_THRESHOLDS["ensemble_confidence"]:.0%}',
             fontsize=14, color='gray', transform=ax3.transAxes)
    y_pos -= 0.08

    # Enhanced processing stats
    ax3.text(0.05, y_pos, '🚀 ENHANCED PROCESSING STATS:', fontsize=14, weight='bold', color='blue', transform=ax3.transAxes)
    y_pos -= 0.04

    stats = [
        f'TTA Variants: {detection_result.get("tta_variants_processed", 0)}',
        f'Raw Detections: {detection_result.get("total_raw_detections", 0)}',
        f'Consensus Detections: {detection_result.get("consensus_detections", 0)}',
        f'OCR Confidence: {detection_result.get("ocr_confidence", 0):.1%}',
        f'Pattern Matches: {detection_result.get("ocr_patterns_found", 0)}',
        f'Processing Stages: {detection_result.get("processing_stages", 0)}'
    ]

    for stat in stats:
        ax3.text(0.1, y_pos, f'• {stat}', fontsize=11, transform=ax3.transAxes)
        y_pos -= 0.03

    y_pos -= 0.02

    # Stage results
    ax3.text(0.05, y_pos, '🔍 STAGE RESULTS:', fontsize=14, weight='bold', color='green', transform=ax3.transAxes)
    y_pos -= 0.04

    stage_results = classification_result.get('stage_results', {})
    for stage_name, stage_data in stage_results.items():
        if isinstance(stage_data, dict):
            key_metric = 'consensus_score' if 'consensus_score' in stage_data else 'combined_confidence' if 'combined_confidence' in stage_data else list(stage_data.keys())[0] if stage_data else 'no_data'
            if key_metric in stage_data:
                value = stage_data[key_metric]
                if isinstance(value, (int, float)):
                    display_value = f"{value:.2f}"
                else:
                    display_value = str(value)
                ax3.text(0.1, y_pos, f'• {stage_name}: {display_value}', fontsize=10, transform=ax3.transAxes)
            else:
                ax3.text(0.1, y_pos, f'• {stage_name}: processed', fontsize=10, transform=ax3.transAxes)
        else:
            ax3.text(0.1, y_pos, f'• {stage_name}: {stage_data}', fontsize=10, transform=ax3.transAxes)
        y_pos -= 0.025

    y_pos -= 0.02

    # Evidence
    evidence = classification_result.get('evidence', [])
    if evidence:
        ax3.text(0.05, y_pos, '🕵️ EVIDENCE:', fontsize=14, weight='bold', color='purple', transform=ax3.transAxes)
        y_pos -= 0.04

        for i, item in enumerate(evidence[:8]):  # Show top 8 evidence items
            ax3.text(0.1, y_pos, f'• {item}', fontsize=9, transform=ax3.transAxes)
            y_pos -= 0.025

    # False negative prevention
    prevention = classification_result.get('false_negative_prevention', [])
    if prevention:
        y_pos -= 0.02
        ax3.text(0.05, y_pos, '🛡️ FALSE NEGATIVE PREVENTION:', fontsize=12, weight='bold', color='red', transform=ax3.transAxes)
        y_pos -= 0.03
        for item in prevention:
            ax3.text(0.1, y_pos, f'• {item}', fontsize=9, transform=ax3.transAxes)
            y_pos -= 0.025

    plt.tight_layout()
    plt.show()

def visualize_detections_enhanced(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Enhanced detection visualization
    """
    vis_image = image.copy()

    if not detections:
        return vis_image

    colors = {
        'AADHAR_NUMBER': (255, 0, 0),     # Red
        'NAME': (0, 255, 0),              # Green
        'ADDRESS': (0, 0, 255),           # Blue
        'DATE_OF_BIRTH': (255, 255, 0),   # Yellow
        'GENDER': (255, 0, 255),          # Magenta
        'ensemble_consensus': (0, 255, 255) # Cyan for consensus
    }

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['box'])
        label = detection['label']
        confidence = detection['confidence']
        model = detection.get('model', 'unknown')

        color = colors.get(label, (128, 128, 128))

        thickness = 3 if model == 'ensemble_consensus' else 2
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

        if model == 'ensemble_consensus':
            consensus_count = detection.get('consensus_count', 1)
            label_text = f"{label}: {confidence:.2f} (C:{consensus_count})"
        else:
            label_text = f"{label}: {confidence:.2f} ({model})"

        font_scale = 0.6
        thickness_text = 1

        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
        cv2.rectangle(vis_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(vis_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_text)

    return vis_image
