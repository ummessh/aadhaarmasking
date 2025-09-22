import numpy as np
import os
from typing import List, Dict
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from src.config import ENSEMBLE_MODELS

class EnhancedEnsembleDetector:
    """
    Advanced ensemble detector implementing research-based improvements
    """

    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.detection_history = []

    def load_ensemble_models(self):
        """
        Load multiple YOLO models for ensemble detection
        Citation: "Model diversity is crucial for effective ensemble learning"
        """
        print("🔄 Loading ensemble models...")
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        if hf_token:
            print("✅ Hugging Face token found.")
        else:
            print("⚠️ Hugging Face token not found. Downloads may be rate-limited.")

        for model_name, config in ENSEMBLE_MODELS.items():
            try:
                model_path = None
                if 'repo_id' in config and 'filename' in config:
                    # Model from Hugging Face
                    model_path = hf_hub_download(
                        repo_id=config['repo_id'],
                        filename=config['filename'],
                        local_dir="./models",
                        token=hf_token
                    )
                elif 'model_name' in config:
                    # Model to be downloaded by YOLO
                    model_path = config['model_name']

                if model_path:
                    self.models[model_name] = YOLO(model_path)
                    self.model_weights[model_name] = config['weight']
                    print(f"✅ {model_name} model loaded")
                else:
                    print(f"⚠️ Configuration for {model_name} is missing 'repo_id'/'filename' or 'model_name'")

            except Exception as e:
                print(f"⚠️ Error loading {model_name}: {e}")

        print(f"🎯 Loaded {len(self.models)} models for ensemble detection")
        return len(self.models) > 0

    def run_ensemble_detection(self, image: np.ndarray) -> Dict:
        """
        Run detection using all models in ensemble - FIXED for shape mismatch issues
        """
        if not self.models:
            print("❌ No models loaded for ensemble detection!")
            return {'detections': [], 'confidence': 0.0}

        all_detections = []
        detection_scores = []

        for model_name, model in self.models.items():
            try:
                config = ENSEMBLE_MODELS.get(model_name, {})
                conf_threshold = config.get('detection_conf', 0.10)
                nms_threshold = config.get('nms_threshold', 0.50)

                if not isinstance(image, np.ndarray):
                    continue

                if len(image.shape) == 3:
                    processed_image = np.ascontiguousarray(image, dtype=np.uint8)
                else:
                    continue

                results = model(processed_image, conf=conf_threshold, iou=nms_threshold, verbose=False)

                if results and len(results) > 0:
                    result = results[0]

                    if result.boxes is not None and len(result.boxes) > 0:
                        try:
                            boxes_tensor = result.boxes.xyxy
                            confidences_tensor = result.boxes.conf
                            classes_tensor = result.boxes.cls

                            if boxes_tensor.numel() > 0:
                                boxes = boxes_tensor.cpu().numpy()
                                confidences = confidences_tensor.cpu().numpy()
                                classes = classes_tensor.cpu().numpy()

                                if (boxes.shape[0] == confidences.shape[0] == classes.shape[0] and
                                    len(boxes.shape) == 2 and boxes.shape[1] == 4):

                                    for i in range(len(boxes)):
                                        try:
                                            box = boxes[i]
                                            conf = confidences[i]
                                            cls = classes[i]

                                            if (len(box) == 4 and
                                                not np.isnan(conf) and not np.isnan(cls) and
                                                conf > 0 and box[0] < box[2] and box[1] < box[3]):

                                                label = model.names.get(int(cls), f"CLASS_{int(cls)}")

                                                detection = {
                                                    'model': model_name,
                                                    'box': box.tolist(),
                                                    'confidence': float(conf),
                                                    'class': int(cls),
                                                    'label': label,
                                                    'weight': self.model_weights[model_name]
                                                }
                                                all_detections.append(detection)
                                        except Exception as e:
                                            print(f"⚠️ Error processing detection {i} from {model_name}: {e}")
                                            continue

                                    if len(confidences) > 0:
                                        model_score = float(np.mean(confidences)) * self.model_weights[model_name]
                                        detection_scores.append(model_score)
                                else:
                                    print(f"⚠️ Invalid detection shapes from {model_name}: boxes={boxes.shape}, conf={confidences.shape}, cls={classes.shape}")

                        except Exception as e:
                            print(f"⚠️ Error extracting detections from {model_name}: {e}")
                            continue

            except Exception as e:
                print(f"⚠️ Detection error with {model_name}: {e}")
                continue

        ensemble_confidence = float(np.mean(detection_scores)) if detection_scores else 0.0

        return {
            'detections': all_detections,
            'confidence': ensemble_confidence,
            'model_count': len([s for s in detection_scores if s > 0])
        }

    def consensus_based_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Advanced NMS using consensus between models
        """
        if not detections:
            return []

        consensus_groups = []
        used_detections = set()

        for i, det1 in enumerate(detections):
            if i in used_detections:
                continue

            group = [det1]
            used_detections.add(i)

            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used_detections:
                    continue

                iou = self.calculate_iou(det1['box'], det2['box'])

                if iou > 0.3 and det1['label'] == det2['label']:
                    group.append(det2)
                    used_detections.add(j)

            consensus_groups.append(group)

        final_detections = []
        for group in consensus_groups:
            if len(group) >= 2:
                consensus_det = self.create_consensus_detection(group)
                final_detections.append(consensus_det)
            elif len(group) == 1 and group[0]['confidence'] > 0.4:
                final_detections.append(group[0])

        return final_detections

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1, x1_2)
        yi1 = max(y1, y1_2)
        xi2 = min(x2, x2_2)
        yi2 = min(y2, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def create_consensus_detection(self, group: List[Dict]) -> Dict:
        """
        Create consensus detection from group
        """
        total_weight = sum(det['weight'] * det['confidence'] for det in group)

        if total_weight == 0:
            return group[0]

        weighted_box = [0, 0, 0, 0]
        for det in group:
            weight = det['weight'] * det['confidence']
            for i in range(4):
                weighted_box[i] += det['box'][i] * weight / total_weight

        consensus_confidence = sum(det['confidence'] * det['weight'] for det in group) / len(group)
        consensus_bonus = min(0.2, 0.05 * (len(group) - 1))
        final_confidence = min(1.0, consensus_confidence + consensus_bonus)

        return {
            'model': 'ensemble_consensus',
            'box': weighted_box,
            'confidence': final_confidence,
            'label': group[0]['label'],
            'consensus_count': len(group),
            'individual_confidences': [det['confidence'] for det in group]
        }
