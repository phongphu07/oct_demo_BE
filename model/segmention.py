from io import BytesIO
import os
import cv2
import numpy as np
from PIL import Image, ImageSequence
import time
from ultralytics import YOLO

class YoloSegmentor:
    def __init__(self):
        self.model = YOLO("./weights/20250430_segment_200epoch_yolov8n_best.pt")

    def predict(self, file_bytes: bytes, filename: str):
        is_tif = filename.lower().endswith((".tif", ".tiff"))
        if is_tif:
            return self.predict_tif_file(file_bytes, filename)
        else:
            return self.predict_single_image(file_bytes, filename)

    def predict_tif_file(self, file_bytes: bytes, filename: str, output_dir: str = "static/results") -> dict:
        image_file = BytesIO(file_bytes)
        image = Image.open(image_file)

        frames = []
        for i in range(20):
            try:
                image.seek(i)
                frames.append((i, image.copy())) 
            except EOFError:
                break

        os.makedirs(output_dir, exist_ok=True)
        results = []

        for idx, page in frames:
            start_time = time.time()

            gray_img = page.convert("L")
            results_yolo = self.model.predict(gray_img, imgsz=1024, conf=0.1, show_labels=False, show_boxes=False)
            boxes_raw = results_yolo[0].boxes.data.cpu().numpy() if results_yolo[0].boxes is not None else []

            annotated = self.textAndContour_segment(page, results_yolo)

            out_filename = f"{os.path.splitext(filename)[0]}_frame_{idx}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, annotated)
            relative_path = os.path.relpath(out_path, "static")

            box_info = []
            label_counts = {}

            for box in boxes_raw:
                x1, y1, x2, y2, conf, cls = box
                label = self.model.names[int(cls)]
                box_info.append({
                    "label": label,
                    "confidence": round(float(conf), 2),
                    "box": [round(x1), round(y1), round(x2), round(y2)]
                })
                label_counts[label] = label_counts.get(label, 0) + 1

            summary_lines = []
            for label in ["lumen", "side_branch"]:
                count = label_counts.get(label, 0)
                if count > 0:
                    summary_lines.append(f"✓ Detected: {count} {label}")
                else:
                    summary_lines.append(f"✓ No {label} detected")

            elapsed = time.time() - start_time
            summary_lines.append(f"✓ Frame: {idx + 1}/{len(frames)}")
            summary_lines.append(f"✓ Processing time: {elapsed:.2f}s")
            summary = "\n".join(summary_lines)

            results.append({
                "frame_index": idx,
                "url": f"/static/{relative_path}",
                "summary": summary,
                "num_boxes": len(box_info),
                "boxes": box_info
            })

        return {
            "type": "tif",
            "frame_count": len(results),
            "frames": results
        }



    def predict_single_image(self, file_bytes: bytes, filename: str, output_dir: str = "static/results") -> dict:
        start_time = time.time()

        image = Image.open(BytesIO(file_bytes))
        os.makedirs(output_dir, exist_ok=True)

        gray_img = image.convert("L")
        results_yolo = self.model.predict(gray_img, imgsz=1024, conf=0.1, show_labels=False, show_boxes=False)
        boxes_raw = results_yolo[0].boxes.data.cpu().numpy() if results_yolo[0].boxes is not None else []

        annotated = self.textAndContour_segment(image, results_yolo)

        out_filename = f"{os.path.splitext(filename)[0]}.jpg"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, annotated)
        relative_path = os.path.relpath(out_path, "static")

        width, height = image.size

        box_info = []
        label_counts = {}

        for box in boxes_raw:
            x1, y1, x2, y2, conf, cls = box
            label = self.model.names[int(cls)]
            box_info.append({
                "label": label,
                "confidence": round(float(conf), 2),
                "box": [round(x1), round(y1), round(x2), round(y2)]
            })
            label_counts[label] = label_counts.get(label, 0) + 1

        summary_lines = []
        for label in ["lumen", "side_branch"]:
            count = label_counts.get(label, 0)
            if count > 0:
                summary_lines.append(f"✓ Detected: {count} {label}")
            else:
                summary_lines.append(f"✓ No {label} detected")

        elapsed = time.time() - start_time
        summary_lines.append(f"✓ Processing time: {elapsed:.2f}s")
        summary = "\n".join(summary_lines)

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
                "summary": summary,
                "num_boxes": len(box_info),
                "boxes": box_info
            }]
        }


    def textAndContour_segment(self, img_path, results):
        img_convert = img_path.convert('RGB')
        img_np = np.array(img_convert)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape

        if results[0].masks is None:
            return results[0].plot(labels=False, boxes=False)

        check_lumen = 0
        for idx, prediction in enumerate(results[0].boxes.xywhn):
            class_id_int = int(results[0].boxes.cls[idx].item())
            poly = results[0].masks.xyn[idx].tolist()
            poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)
            poly *= [w, h]

            if class_id_int == 0 and check_lumen == 0:
                cv2.polylines(img, [poly.astype('int')], True, (255, 0, 0), 2)
                check_lumen += 1
            elif class_id_int == 1:
                cv2.polylines(img, [poly.astype('int')], True, (0, 255, 0), 2)

        return img
