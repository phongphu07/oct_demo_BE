import os
import io
import cv2
import numpy as np
from PIL import Image, ImageSequence
from uuid import uuid4
from ultralytics import YOLO

LCA_CLASSES = ['5', '6', '7', '8', '9', '9a', '10', '10a', '11', '12', '12a', '13', '14', '14a', '15', '12b', '14b']
RCA_CLASSES = ['1', '2', '3', '4', '16', '16a', '16b', '16c']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static", "results")
os.makedirs(STATIC_DIR, exist_ok=True)

class AngioFFR:
    def __init__(self):
        self.cls_model_path = os.path.join(BASE_DIR, "weights", "cls_400epoch_50patience_best.pt")
        self.seg_lca_path = os.path.join(BASE_DIR, "weights", "241003_yolov8x_lca_dropout05_best.pt")
        self.seg_rca_path = os.path.join(BASE_DIR, "weights", "241003_yolov8x_rca_dropout05_best.pt")

        self.classifier = YOLO(self.cls_model_path)
        self.segmentor_lca = YOLO(self.seg_lca_path)
        self.segmentor_rca = YOLO(self.seg_rca_path)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def predict(self, file: bytes, filename: str):
        image_pil = Image.open(io.BytesIO(file)).convert("RGB")
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        enhanced = self.enhance_contrast(image_bgr)

        # Classification
        cls_result = self.classifier.predict(enhanced, conf=0.5, verbose=False)
        probs = cls_result[0].probs.data.tolist()
        vessel_type = 'LCA' if probs[0] > probs[1] else 'RCA'

        # Segmentation
        segmentor = self.segmentor_lca if vessel_type == 'LCA' else self.segmentor_rca
        seg_result = segmentor.predict(enhanced, save=False, conf=0.25, verbose=False)[0]

        # Annotate and save
        annotated = seg_result.plot(labels=True, boxes=True, probs=False)
        output_filename = f"seg_{vessel_type}_{filename}"
        output_path = os.path.join(STATIC_DIR, output_filename)
        cv2.imwrite(output_path, annotated)

        # Box info + class count
        boxes_info = []
        class_count = {}

        for box in seg_result.boxes:
            cls_id = int(box.cls[0])
            class_name = seg_result.names[cls_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            boxes_info.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "box": [round(coord, 2) for coord in xyxy]
            })
            class_count[class_name] = class_count.get(class_name, 0) + 1

        summary = ", ".join(f"{k}: {v}" for k, v in class_count.items()) if boxes_info else "Không có"
        height, width = annotated.shape[:2]

        return {
            "vessel_type": vessel_type,
            "output_path": f"/static/results/{output_filename}",
            "num_boxes": len(boxes_info),
            "class_distribution": class_count,
            "summary": summary,
            "boxes_info": boxes_info,
            "width": width,
            "height": height
        }
    
    def predict_tif_file(self, file: bytes, filename: str, output_dir: str):
        image_pil = Image.open(io.BytesIO(file))
        frames = []
        idx = 0

        for page in ImageSequence.Iterator(image_pil):
            # Chuyển frame sang BGR
            rgb = page.convert("RGB")
            image_np = np.array(rgb)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            enhanced = self.enhance_contrast(image_bgr)

            # Classification
            cls_result = self.classifier.predict(enhanced, conf=0.5, verbose=False)
            probs = cls_result[0].probs.data.tolist()
            vessel_type = 'LCA' if probs[0] > probs[1] else 'RCA'
            segmentor = self.segmentor_lca if vessel_type == 'LCA' else self.segmentor_rca

            # Segmentation
            seg_result = segmentor.predict(enhanced, save=False, conf=0.25, verbose=False)[0]
            annotated = seg_result.plot(labels=True, boxes=True, probs=False)

            # Save image
            output_filename = f"seg_{vessel_type}_{idx}_{filename.replace('.tif', '.png')}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, annotated)

            # Box info + class count
            boxes_info = []
            class_count = {}

            for box in seg_result.boxes:
                cls_id = int(box.cls[0])
                class_name = seg_result.names[cls_id]
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                boxes_info.append({
                    "class": class_name,
                    "confidence": round(confidence, 2),
                    "box": [round(coord, 2) for coord in xyxy]
                })
                class_count[class_name] = class_count.get(class_name, 0) + 1

            summary = (
                ", ".join(f"{k}: {v}" for k, v in class_count.items())
                if boxes_info else "No objects detected"
            )

            height, width = annotated.shape[:2]

            frames.append({
                "url": f"/static/results/{output_filename}",
                "boxes": boxes_info,
                "num_boxes": len(boxes_info),
                "summary": summary,
                "class_distribution": class_count,
                "image_size": {
                    "width": width,
                    "height": height
                },
                "vessel_type": vessel_type
            })

            idx += 1

        return {
            "type": "tif",
            "frame_count": len(frames),
            "frames": frames
        }

