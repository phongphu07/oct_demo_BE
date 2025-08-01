import io
import numpy as np
import cv2
import os
from PIL import Image, ImageSequence
from sympy import content
from ultralytics import YOLO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

class YoloDetection:
    def __init__(self, model_path='./weights/train_250524_yolov8m_500epoch_imgsz1024.pt'):
        self.model = YOLO(model_path)

    def predict_single_image(
        self, 
        file: bytes, 
        filename: str, 
        output_path: str,
        frame_idx: int = None, 
        total_frames: int = None
    ):
        import time
        start_time = time.time()

        img_pil = Image.open(io.BytesIO(file)).convert("RGB")
        # img_pil = Image.open(io.BytesIO(content)).convert("RGB")
        width, height = img_pil.size

        image_np = np.array(img_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = self.model.predict(img_pil, imgsz=1024, conf=0.25, show_labels=False)

        annotated_img = self.text_and_contour(image_bgr, results, width, height)
        cv2.imwrite(output_path, annotated_img)

        boxes_info = []
        class_counts = Counter()
        for idx, prediction in enumerate(results[0].boxes.xywhn):
            cls_id = int(results[0].boxes.cls[idx].item())
            x, y, w, h = prediction.tolist()
            boxes_info.append({
                "class": cls_id,
                "x_center_norm": x,
                "y_center_norm": y,
                "width_norm": w,
                "height_norm": h
            })
            class_counts[cls_id] += 1

        class_names = self.model.names 

        summary_lines = []

        for class_id, class_name in class_names.items():
            count = class_counts.get(class_id, 0)
            if count > 0:
                summary_lines.append(f"✓ Detected: {count} {class_name}")
            else:
                summary_lines.append(f"✓ No {class_name} detected")

        if frame_idx is not None and total_frames is not None:
            summary_lines.append(f"✓ Frame: {frame_idx + 1}/{total_frames}")

        elapsed = time.time() - start_time
        summary_lines.append(f"✓ Processing time: {elapsed:.2f}s")

        return {
            "output_path": output_path,
            "boxes": boxes_info,
            "image_size": {
                "width": width,
                "height": height
            },
            "summary": "\n".join(summary_lines),
            "num_boxes": len(boxes_info)
        }

    
    def predict_tif_file(self, file: bytes, filename: str, output_dir: str = "static/results") -> list[dict]:
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(io.BytesIO(file))
        frames = []

        for i, frame in enumerate(ImageSequence.Iterator(image)):
            if i >= 20:
                break
            frame_rgb = frame.convert("RGB")
            frames.append((i, frame_rgb))

        def process_frame(i_and_frame):
            i, frame_rgb = i_and_frame

            frame_bytes = io.BytesIO()
            frame_rgb.save(frame_bytes, format="PNG")
            frame_bytes.seek(0)

            frame_filename = f"{os.path.splitext(filename)[0]}_frame_{i}.png"
            output_path = os.path.join(output_dir, frame_filename)

            result = self.predict_single_image(
                frame_bytes.getvalue(),
                frame_filename,
                output_path,
                frame_idx=i,
                total_frames=len(frames)
            )

            relative_path = os.path.relpath(output_path, STATIC_DIR)
            return {
                "frame_index": i,
                "url": f"/static/{relative_path}",
                "summary": result["summary"],
                "num_boxes": result["num_boxes"],
            }

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_frame, frames))

        results.sort(key=lambda x: x["frame_index"])
        return results
    
    def text_and_contour(self, image_bgr, results, width, height):
        dh, dw, _ = image_bgr.shape

        boxes = []
        for idx, prediction in enumerate(results[0].boxes.xywhn):
            cl = int(results[0].boxes.cls[idx].item())
            x, y, w, h = prediction.tolist()
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)
            boxes.append((cl, max(0, l), max(0, t), min(dw - 1, r), min(dh - 1, b)))

        for cl, x1, y1, x2, y2 in boxes:
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (0, 255, 0) if cl == 0 else (255, 0, 255)
            cv2.drawContours(image_bgr[y1:y2, x1:x2], contours, -1, color, 2 if cl == 0 else 10)

        return image_bgr
