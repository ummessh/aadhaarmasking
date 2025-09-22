import torch

# Model ensemble configuration
ENSEMBLE_MODELS = {
    'yolov8_nano': {
        'repo_id': 'arnabdhar/YOLOv8-nano-aadhar-card',
        'filename': 'model.pt',
        'weight': 0.4,  # Proven model gets substantial weight
        'detection_conf': 0.10,  # Lower for ensemble diversity
        'nms_threshold': 0.45
    },
    'yolov8_medium': {
        'model_name': 'yolov8m.pt',  # Medium model for robustness
        'weight': 0.35,
        'detection_conf': 0.10,
        'nms_threshold': 0.50
    },
    'yolov8_small': {
        'model_name': 'yolov8s.pt',  # Larger model for better accuracy
        'weight': 0.25,  # Additional accuracy from larger model
        'detection_conf': 0.08,  # Very sensitive detection
        'nms_threshold': 0.55
    }
}

# Enhanced classification thresholds
CLASSIFICATION_THRESHOLDS = {
    'ensemble_confidence': 0.35,  # Lower due to ensemble boost
    'min_field_count': 2,         # At least 2 fields detected
    'aadhaar_pattern_bonus': 0.15, # OCR pattern match bonus
    'field_consensus_threshold': 0.6,  # Agreement between models
    'min_template_matches': 3,  # Minimum number of visual templates to match for a significant confidence boost
    'template_matching_weight': 0.6  # The weight given to template matching in the final score
}

# Test-Time Augmentation settings
TTA_AUGMENTATIONS = {
    'enabled': True,
    'scales': [640, 832, 1024],  # Multi-scale inference
    'flips': [False, True],       # Horizontal flip
    'rotations': [0, 90, 180, 270],  # Rotation variants
    'brightness': [0.8, 1.0, 1.2],  # Brightness variants
}

# Enhanced field patterns for validation
# Translations sourced from Google Translate and regional language experts.
ENHANCED_AADHAAR_PATTERNS = {
    'aadhaar_number': [
        r'\b\d{4}\s+\d{4}\s+\d{4}\b',      # XXXX XXXX XXXX
        r'\b\d{12}\b',                      # XXXXXXXXXXXX
        r'\b\d{4}-\d{4}-\d{4}\b',          # XXXX-XXXX-XXXX
        r'\b\d{4}\s*\d{4}\s*\d{4}\b',      # Flexible spacing
    ],
    'government_indicators': [
        # English
        r'government\s+of\s+india',
        r'unique\s+identification',
        r'aadhaar',
        # Devanagari Script (Hindi, Marathi, Nepali, Konkani, etc.)
        r'भारत\s+सरकार',
        r'आधार',
        # Tamil
        r'இந்திய\s+அரசு',
        # Telugu
        r'భారత\s+ప్రభుత్వం',
        # Kannada
        r'ಭಾರತ\s+ಸರ್ಕಾರ',
        # Bengali / Assamese
        r'ভারত\s+সরকার',
        # Malayalam
        r'ഭാരത\s+സർക്കാർ',
        # Urdu
        r'حکومتِ\s+ہند',
    ],
    'date_patterns': [
        r'\b\d{2}/\d{2}/\d{4}\b',
        r'\b\d{2}-\d{2}-\d{4}\b',
        r'\b\d{2}\.\d{2}\.\d{4}\b'
    ],
    'gender_patterns': [
        r'\b(male|female|पुरुष|महिला|म|फ)\b',
        # Tamil
        r'\b(ஆண்|பெண்)\b',
        # Telugu
        r'\b(పురుషుడు|స్త్రీ)\b',
        # Kannada
        r'\b(ಗಂಡು|ಹೆಣ್ಣು)\b',
    ]
}

# Set device for optimal performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
