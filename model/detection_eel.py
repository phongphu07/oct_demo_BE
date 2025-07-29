from io import BytesIO
import cv2
from ultralytics import YOLO
from PIL import Image, ImageSequence
import os
import numpy as np
from operator import add, mul

class YoloDetection_EEL:
    def __init__(self):
        self.model = YOLO('./weights/eel_2093img_100epoch.pt')
        self.model_keypoint = YOLO('./weights/train_240611_eel_keypoint_new_LUT.pt')
        self.output_dir = "static/results"
        os.makedirs(self.output_dir, exist_ok=True)

    def predict(self, file, filename, alpha=0.05):
        im = Image.open(BytesIO(file))
        is_tif = filename.lower().endswith((".tif", ".tiff"))
        frames = list(ImageSequence.Iterator(im)) if is_tif else [im]

        results = []

        for j, page in enumerate(frames):
            results_segment = self.model.predict(page, imgsz=1024, conf=0.5, show_labels=False, show_boxes=False)
            annotated_segment = self.textAndContour_segment_calcium(page, results_segment)

            results_keypoint = self.model_keypoint.predict(
                annotated_segment, imgsz=1024, conf=0.5, show_labels=False, show_boxes=False
            )
            annotated_frame = self.textAndContour_segment_eel_keypoint(page, results_segment, results_keypoint, alpha)

            out_filename = f"{os.path.splitext(filename)[0]}_eel_frame_{j:03d}.jpg"
            out_path = os.path.join(self.output_dir, out_filename)
            cv2.imwrite(out_path, annotated_frame)
            relative_path = os.path.relpath(out_path, "static")

            num_eel = len(results_segment[0].masks) if results_segment[0].masks is not None else 0

            results.append({
                "frame_index": j,
                "url": f"/static/{relative_path}",
                "summary": f"{num_eel} EEL regions detected",
                "boxes": [],
                "class_distribution": {}
            })

        return {
            "type": "tif" if is_tif else "image",
            "frame_count": len(results),
            "frames": results
        }

    def textAndContour_segment_calcium(self, img_path, results):
        img_convert = img_path.convert('RGB')
        img_np = np.array(img_convert)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        labels_mask = results[0].masks

        if labels_mask is None:
            return results[0].plot(labels=False, boxes=False)

        for idx, prediction in enumerate(results[0].boxes.xywhn):
            poly = results[0].masks.xyn[idx].tolist()
            poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            poly *= [w, h]
            cv2.polylines(img, [poly.astype('int')], True, (255, 0, 0), 2)
        return img

    def is_point_between(self, point, point1, point2):
        min_x = min(point1[0], point2[0])
        max_x = max(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        max_y = max(point1[1], point2[1])
        return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

    def textAndContour_segment_eel_keypoint(self, img_path, results, results_keypoint, alpha):
        img_convert = img_path.convert('RGB')
        img_np = np.array(img_convert)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        labels_mask = results[0].masks

        if labels_mask is None:
            return results[0].plot(labels=False, boxes=False)

        center_point = [0.5, 0.5]
        final_arr = []
        big_list = []
        const_modify = [alpha, alpha]

        lt_list = list(map(mul, [-1, -1], const_modify))
        rt_list = list(map(mul, [1, -1], const_modify))
        rb_list = list(map(mul, [1, 1], const_modify))
        lb_list = list(map(mul, [-1, 1], const_modify))

        for idx, prediction in enumerate(results[0].boxes.xywhn):
            poly = results[0].masks.xyn[idx].tolist()

            for i in range(len(results_keypoint[0].keypoints.xyn)):
                point_list = results_keypoint[0].keypoints.xyn[i].tolist()
                if len(point_list) < 3:
                    continue  # tránh lỗi nếu thiếu keypoint
                big_list.append(point_list)

            poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

            for point in poly:
                flag = 1
                if 0.45 <= point[0] <= 0.55 and 0.45 <= point[1] <= 0.55:
                    continue
                for point_list in big_list:
                    pt_lt_1 = list(map(add, point_list[1], lt_list))
                    pt_rt_1 = list(map(add, point_list[1], rt_list))
                    pt_lb_1 = list(map(add, point_list[1], lb_list))
                    pt_rb_1 = list(map(add, point_list[1], rb_list))
                    pt_lt_2 = list(map(add, point_list[2], lt_list))
                    pt_rt_2 = list(map(add, point_list[2], rt_list))
                    pt_lb_2 = list(map(add, point_list[2], lb_list))
                    pt_rb_2 = list(map(add, point_list[2], rb_list))
                    if (
                        self.is_point_between(point, center_point, point_list[1]) or
                        self.is_point_between(point, center_point, point_list[2]) or
                        self.is_point_between(point, center_point, pt_lt_1) or
                        self.is_point_between(point, center_point, pt_rt_1) or
                        self.is_point_between(point, center_point, pt_lb_1) or
                        self.is_point_between(point, center_point, pt_rb_1) or
                        self.is_point_between(point, center_point, pt_lt_2) or
                        self.is_point_between(point, center_point, pt_rt_2) or
                        self.is_point_between(point, center_point, pt_lb_2) or
                        self.is_point_between(point, center_point, pt_rb_2)
                    ):
                        flag = 0
                        break
                if flag == 1:
                    final_arr.append(point)

        final_arr = np.asarray(final_arr, dtype=np.float32).reshape(-1, 2)
        final_arr *= [w, h]
        final_arr = final_arr.astype('int')
        for point in final_arr:
            cv2.circle(img, point, radius=2, color=(255, 0, 0), thickness=-1)
        return img