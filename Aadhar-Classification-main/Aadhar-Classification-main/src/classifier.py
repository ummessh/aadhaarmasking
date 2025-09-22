import numpy as np
import re
import cv2
import os
import logging
from typing import List, Dict
from PIL import Image
import imagehash
from src.config import CLASSIFICATION_THRESHOLDS

# Configure logging
logger = logging.getLogger(__name__)

class FalseNegativePreventionClassifier:
    """
    Advanced classifier designed to achieve 100% accuracy on ideal Aadhaar cards
    """

    def __init__(self):
        self.classification_stages = []
        self.confidence_boosters = []
        self.templates = self.load_templates('templates')

    def load_templates(self, template_dir: str) -> Dict:
        """Load all template images from a directory using OpenCV."""
        templates = {}
        if not os.path.isdir(template_dir):
            logger.warning(f"Template directory not found: {template_dir}")
            return templates
        for filename in os.listdir(template_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(template_dir, filename)
                template_name = os.path.splitext(filename)[0]
                try:
                    template_image = cv2.imread(path)
                    if template_image is not None:
                        templates[template_name] = template_image
                    else:
                        logger.warning(f"Could not load template image: {filename}")
                except Exception as e:
                    logger.error(f"Error loading template {filename}: {e}")
        logger.info(f"Loaded {len(templates)} templates from {template_dir}")
        return templates

    def multi_stage_classification(self, detections: List[Dict], ocr_data: Dict, image: np.ndarray) -> Dict:
        """
        Multi-stage classification to prevent false negatives
        """
        analysis = {
            'is_aadhaar': False,
            'confidence': 0.0,
            'stage_results': {},
            'evidence': [],
            'detected_fields': [],
            'false_negative_prevention': []
        }
        logger.info("Starting multi-stage classification.")

        # Stage 1: YOLO Detection Analysis
        stage1_result = self.analyze_yolo_detections(detections)
        analysis['stage_results']['yolo'] = stage1_result
        logger.info(f"YOLO analysis complete. Score: {stage1_result.get('consensus_score', 0.0):.2f}, Fields: {stage1_result.get('fields', [])}")

        # Stage 2: OCR Pattern Validation
        stage2_result = self.analyze_ocr_patterns(ocr_data)
        analysis['stage_results']['ocr'] = stage2_result
        logger.info(f"OCR analysis complete. Strength: {stage2_result.get('aadhaar_strength', 0.0):.2f}, Fields: {stage2_result.get('fields', [])}")

        # Stage 3: Combined Evidence Assessment
        stage3_result = self.combined_evidence_analysis(stage1_result, stage2_result)
        analysis['stage_results']['combined'] = stage3_result
        logger.info(f"Combined evidence analysis complete. Confidence: {stage3_result.get('combined_confidence', 0.0):.2f}")

        # Stage 4: Visual Template Matching
        stage4_result = self.analyze_visual_templates(image)
        analysis['stage_results']['visual'] = stage4_result
        logger.info(f"Visual template analysis complete. Templates found: {list(stage4_result.get('templates_found', {}).keys())}")

        # Stage 5: False Negative Prevention Checks
        stage5_result = self.false_negative_prevention_checks(detections, ocr_data, image.shape)
        analysis['stage_results']['prevention'] = stage5_result
        logger.info(f"False negative prevention checks complete. Rescue confidence: {stage5_result.get('rescue_confidence', 0.0):.2f}")

        # Final Classification with Multiple Fallbacks
        final_confidence = self.calculate_final_confidence(
            stage1_result, stage2_result, stage3_result, stage4_result, stage5_result
        )
        logger.info(f"Final confidence calculated: {final_confidence:.2f}")

        analysis['confidence'] = float(final_confidence)
        analysis['is_aadhaar'] = bool(final_confidence >= CLASSIFICATION_THRESHOLDS['ensemble_confidence'])

        # Collect all evidence
        analysis['evidence'] = self.collect_evidence(stage1_result, stage2_result, stage3_result, stage4_result, stage5_result)
        analysis['detected_fields'] = list(set(stage1_result.get('fields', []) + stage2_result.get('fields', [])))
        analysis['false_negative_prevention'] = stage4_result.get('prevention_triggers', [])

        return analysis

    def analyze_yolo_detections(self, detections: List[Dict]) -> Dict:
        """
        Analyze YOLO detection results with consensus scoring
        """
        result = {
            'detection_count': len(detections),
            'fields': [],
            'confidence_scores': [],
            'consensus_score': 0.0,
            'field_coverage': 0.0
        }

        if not detections:
            return result

        field_mappings = {
            'AADHAR_NUMBER': ['AADHAR', 'NUMBER', 'ID'],
            'NAME': ['NAME', 'नाम'],
            'ADDRESS': ['ADDRESS', 'पता'],
            'DATE_OF_BIRTH': ['DATE', 'BIRTH', 'DOB', 'जन्म'],
            'GENDER': ['GENDER', 'लिंग', 'MALE', 'FEMALE']
        }

        detected_fields = set()
        confidence_scores = []

        for detection in detections:
            label = detection['label'].upper()
            confidence = detection['confidence']
            confidence_scores.append(confidence)

            for standard_field, variants in field_mappings.items():
                if any(variant in label for variant in variants):
                    detected_fields.add(standard_field)
                    break
            else:
                detected_fields.add(label)

        result['fields'] = list(detected_fields)
        result['confidence_scores'] = confidence_scores
        result['consensus_score'] = float(np.mean(confidence_scores)) if confidence_scores else 0.0

        essential_fields = ['AADHAR_NUMBER', 'NAME']
        found_essential = sum(1 for field in essential_fields if field in detected_fields)
        result['field_coverage'] = found_essential / len(essential_fields)

        return result

    def analyze_ocr_patterns(self, ocr_data: Dict) -> Dict:
        """
        Analyze OCR patterns for Aadhaar indicators
        """
        result = {
            'pattern_matches': {},
            'confidence_score': ocr_data.get('confidence', 0.0),
            'text_length': len(ocr_data.get('text', '')),
            'fields': [],
            'aadhaar_strength': 0.0
        }

        patterns = ocr_data.get('patterns', {})
        text = ocr_data.get('text', '').lower()

        for pattern_type, matches in patterns.items():
            if matches:
                result['pattern_matches'][pattern_type] = len(matches)
                if pattern_type == 'aadhaar_number':
                    result['fields'].append('AADHAR_NUMBER')
                elif pattern_type == 'date_patterns':
                    result['fields'].append('DATE_OF_BIRTH')
                elif pattern_type == 'gender_patterns':
                    result['fields'].append('GENDER')
                elif pattern_type == 'government_indicators':
                    result['fields'].append('GOVERNMENT_INDICATOR')

        strength_indicators = [
            ('aadhaar_number', 0.5),
            ('government_indicators', 0.3),
            ('date_patterns', 0.1),
            ('gender_patterns', 0.1)
        ]

        total_strength = 0.0
        for indicator, weight in strength_indicators:
            if indicator in result['pattern_matches']:
                total_strength += weight * min(1.0, result['pattern_matches'][indicator])

        result['aadhaar_strength'] = min(1.0, total_strength)
        return result

    def combined_evidence_analysis(self, yolo_result: Dict, ocr_result: Dict) -> Dict:
        """
        Combine YOLO and OCR evidence for robust classification
        """
        result = {
            'combined_confidence': 0.0,
            'field_consensus': {},
            'evidence_strength': 0.0,
            'multi_modal_validation': False
        }

        yolo_fields = set(yolo_result.get('fields', []))
        ocr_fields = set(ocr_result.get('fields', []))

        all_fields = yolo_fields.union(ocr_fields)
        consensus_fields = yolo_fields.intersection(ocr_fields)

        for field in all_fields:
            consensus_score = 0.0
            if field in consensus_fields:
                consensus_score = 1.0
            elif field in yolo_fields:
                consensus_score = 0.6
            elif field in ocr_fields:
                consensus_score = 0.4
            result['field_consensus'][field] = consensus_score

        yolo_weight = 0.6
        ocr_weight = 0.4
        yolo_conf = yolo_result.get('consensus_score', 0.0)
        ocr_conf = ocr_result.get('aadhaar_strength', 0.0)
        result['combined_confidence'] = (yolo_conf * yolo_weight) + (ocr_conf * ocr_weight)

        has_aadhaar_number = ('AADHAR_NUMBER' in yolo_fields) or ('AADHAR_NUMBER' in ocr_fields)
        has_additional_field = len(all_fields) >= 2
        result['multi_modal_validation'] = bool(has_aadhaar_number and has_additional_field)

        field_strength = len(consensus_fields) * 0.3 + len(all_fields) * 0.1
        confidence_strength = result['combined_confidence'] * 0.6
        result['evidence_strength'] = min(1.0, field_strength + confidence_strength)

        return result

    def false_negative_prevention_checks(self, detections: List[Dict], ocr_data: Dict, image_shape: tuple) -> Dict:
        """
        Advanced checks to prevent false negatives on genuine Aadhaar cards
        """
        result = {
            'prevention_triggers': [],
            'rescue_confidence': 0.0,
            'quality_indicators': {},
            'should_rescue': False
        }

        height, width = image_shape[:2]
        total_pixels = height * width

        result['quality_indicators'] = {
            'resolution': 'high' if total_pixels > 400000 else 'medium' if total_pixels > 100000 else 'low',
            'aspect_ratio': width / height,
            'size_score': min(1.0, total_pixels / 500000)
        }

        weak_signals = []
        text = ocr_data.get('text', '').lower()
        if any(indicator in text for indicator in ['aadhaar', 'आधार', 'government', 'india']):
            weak_signals.append('government_text_present')
        if re.search(r'\d{12}|\d{4}\s+\d{4}\s+\d{4}', text):
            weak_signals.append('number_pattern_present')

        if detections:
            detection_confidences = [d['confidence'] for d in detections]
            if any(conf > 0.2 for conf in detection_confidences):
                weak_signals.append('yolo_weak_detection')
            if len(detections) >= 2:
                weak_signals.append('multiple_detections')

        rescue_score = 0.0
        if result['quality_indicators']['resolution'] == 'high':
            rescue_score += 0.2
        if 0.4 < result['quality_indicators']['aspect_ratio'] < 2.5:
            rescue_score += 0.1
        rescue_score += len(weak_signals) * 0.15
        if any(pattern in text for pattern in ['aadhaar', 'आधार']) and len(weak_signals) >= 2:
            rescue_score += 0.3
            result['prevention_triggers'].append('aadhaar_text_with_signals')

        result['rescue_confidence'] = float(min(1.0, rescue_score))
        result['should_rescue'] = bool(rescue_score >= 0.4)
        if result['should_rescue']:
            result['prevention_triggers'].append('false_negative_prevention_activated')

        return result

    def analyze_visual_templates(self, image: np.ndarray) -> Dict:
        """Analyze image for visual templates using scale-invariant template matching."""
        result = {'templates_found': {}}
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_h, image_w = gray_image.shape[:2]

        for name, template_image in self.templates.items():
            if template_image is None:
                continue
            
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            template_h, template_w = template_gray.shape[:2]
            
            best_match_score = 0.0

            # Iterate over a range of scales
            for scale in np.linspace(0.5, 1.5, 15):
                resized_w, resized_h = int(template_w * scale), int(template_h * scale)
                
                if resized_w > image_w or resized_h > image_h:
                    continue

                resized_template = cv2.resize(template_gray, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
                
                res = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val > best_match_score:
                    best_match_score = max_val

            threshold = 0.65
            logger.info(f"Template '{name}' best match score: {best_match_score:.2f}")
            if best_match_score >= threshold:
                result['templates_found'][name] = best_match_score
                logger.info(f"Template match found: {name} with confidence {best_match_score:.2f}")
        return result

    def calculate_final_confidence(self, stage1: Dict, stage2: Dict, stage3: Dict, stage4: Dict, stage5: Dict) -> float:
        """
        Calculate final confidence with a weighted score for template matching.
        """
        # Prioritize combined evidence from YOLO and OCR
        evidence_confidence = stage3.get('combined_confidence', 0.0)
        
        # Use template matching as a supplemental score
        num_templates_found = len(stage4.get('templates_found', {}))
        template_boost = 0.15 * num_templates_found  # Add a small boost for each template match

        # The base confidence is now primarily from evidence
        base_confidence = evidence_confidence

        # Apply a boost from templates
        boosted_confidence = base_confidence + template_boost

        # Apply rescue boost if other signals are weak but present
        rescue_confidence = stage5.get('rescue_confidence', 0.0)
        rescue_boost = min(0.25, rescue_confidence * 0.5) if stage5.get('should_rescue', False) else 0.0
        
        final_confidence = boosted_confidence + rescue_boost
        logger.info(f"Final confidence breakdown: Evidence={evidence_confidence:.2f}, TemplateBoost={template_boost:.2f}, RescueBoost={rescue_boost:.2f}")
        return min(1.0, final_confidence)

    def collect_evidence(self, stage1: Dict, stage2: Dict, stage3: Dict, stage4: Dict, stage5: Dict) -> List[str]:
        """
        Collect all evidence for the classification decision
        """
        evidence = []
        
        # Detailed YOLO Evidence
        detected_fields = stage1.get('fields', [])
        if detected_fields:
            evidence.append(f"YOLO detected fields: {', '.join(detected_fields)}")
        
        # Detailed OCR Evidence
        patterns = stage2.get('pattern_matches', {})
        if patterns:
            ocr_evidence = "OCR found patterns: " + ", ".join([f"{count}x {ptype}" for ptype, count in patterns.items()])
            evidence.append(ocr_evidence)

        # Combined Evidence
        consensus_fields = [field for field, score in stage3.get('field_consensus', {}).items() if score == 1.0]
        if consensus_fields:
            evidence.append(f"Multi-modal consensus on fields: {', '.join(consensus_fields)}")
        
        if stage3.get('multi_modal_validation', False):
            evidence.append("Multi-modal validation passed (Aadhaar number and other fields found)")

        # Visual Template Evidence
        templates_found = stage4.get('templates_found', {})
        if templates_found:
            template_evidence = "Visual templates found: " + ", ".join([f"{name} (conf: {conf:.2f})" for name, conf in templates_found.items()])
            evidence.append(template_evidence)

        # Prevention Triggers
        triggers = stage5.get('prevention_triggers', [])
        if triggers:
            evidence.append(f"False negative prevention triggered by: {', '.join(triggers)}")
            
        return evidence
