import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

class YoloSegmentor_Calcium:
    def __init__(self):
        self.model = YOLO('./weights/240403_calcium_2022_30epoch_best.pt')
        self.output_dir = "static/results"
        os.makedirs(self.output_dir, exist_ok=True)

    def predict(self, file_bytes: bytes, filename: str):
        is_tif = filename.lower().endswith((".tif", ".tiff"))
        if is_tif:
            return self.predict_tif_file(file_bytes, filename)
        else:
            return self.predict_single_image(file_bytes, filename)

    def predict_tif_file(self, file_bytes: bytes, filename: str) -> dict:
        image = Image.open(BytesIO(file_bytes))
        frames = []
        for i in range(20):
            try:
                image.seek(i)
                frames.append((i, image.copy()))
            except EOFError:
                break

        results = []

        for idx, page in frames:
            gray_img = page.convert("L")
            results_yolo = self.model.predict(gray_img, imgsz=1024, conf=0.1, show_labels=False, show_boxes=False)
            annotated = self.textAndContour_segment_calcium(page, results_yolo)
            num_calcium = len(results_yolo[0].masks) if results_yolo[0].masks is not None else 0
            out_filename = f"{os.path.splitext(filename)[0]}_frame_{idx}.jpg"
            out_path = os.path.join(self.output_dir, out_filename)
            cv2.imwrite(out_path, annotated)
            relative_path = os.path.relpath(out_path, "static")

            results.append({
                "frame_index": idx,
                "url": f"/static/{relative_path}",
                "summary": f"{num_calcium} calcium regions detected",
                "num_boxes": 0,
                "boxes": []
            })

        return {
            "type": "tif",
            "frame_count": len(results),
            "frames": results
        }

    def predict_single_image(self, file_bytes: bytes, filename: str) -> dict:
        image = Image.open(BytesIO(file_bytes))
        gray_img = image.convert("L")
        results_yolo = self.model.predict(gray_img, imgsz=1024, conf=0.1, show_labels=False, show_boxes=False)
        annotated = self.textAndContour_segment_calcium(image, results_yolo)
        num_calcium = len(results_yolo[0].masks) if results_yolo[0].masks is not None else 0
        out_filename = f"{os.path.splitext(filename)[0]}.jpg"
        out_path = os.path.join(self.output_dir, out_filename)
        cv2.imwrite(out_path, annotated)
        relative_path = os.path.relpath(out_path, "static")

        width, height = image.size

        return {
            "type": "image",
            "frame_count": 1,
            "image_size": {
                "width": width,
                "height": height
            },
            "frames": [{
                "frame_index": 0,
                "url": f"/static/{relative_path}",
                "summary": f"{num_calcium} calcium regions detected",
                "num_boxes": 0,
                "boxes": []
            }]
        }

    def textAndContour_segment_calcium(self, img_path, results):
        img_convert = img_path.convert('RGB')
        img_np = np.array(img_convert)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        labels_mask = results[0].masks
        if labels_mask is None:
            return cv2.cvtColor(np.array(img_path.convert("RGB")), cv2.COLOR_RGB2BGR)

        for idx, prediction in enumerate(results[0].boxes.xywhn):
            poly = results[0].masks.xyn[idx].tolist()
            poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            poly *= [w, h]
            cv2.polylines(img, [poly.astype(np.int32)], True, (255, 0, 0), 2)

        return img
