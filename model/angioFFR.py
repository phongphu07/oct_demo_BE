from collections import Counter
import cv2
import os
import numpy as np
import pydicom as dicom
from ultralytics import YOLO
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import load
import pydicom

class CombinedPipeline:
    def __init__(self):
        self.classification_model = YOLO('./weights/cls_80epoch.pt')
        self.segmentation_model = YOLO('./weights/anatomic_segmentation_0708.pt')
        self.lesion_detection_model = YOLO('./weights/lesion_detection_15epoch_297.pt')
        self.regression_model = load('./weights/rf_full_pipeline_240808.joblib')

    def predict(self, dicom_path: str, excel_path: str, output_dir: str) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        best_frame = self.extract_best_frame(dicom_path)
        if best_frame is None:
            return {"error": "No suitable high-quality frame found."}

        segmented = self.segment_frame(best_frame)
        cv2.imwrite(os.path.join(output_dir, "best_segmented_frame.png"), segmented)

        lesion_results = self.detect_lesions(segmented, output_dir)
        self.update_excel(excel_path, lesion_results)

        data = self.prepare_data_for_regression(excel_path)
        prediction = self.regression_model.predict(data)

        return {
            "prediction": prediction.tolist(),
            "lesions": lesion_results
        }

    def extract_best_frame(self, dicom_path: str):
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array

        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array
        elif ds.PhotometricInterpretation == "YBR_FULL":
            pixel_array = np.frombuffer(ds.PixelData, dtype=np.uint8).reshape(ds.Rows, ds.Columns, 3)

        pixel_array = pixel_array.astype(np.uint8)

        best_frame = None
        best_conf = -1

        for i in range(pixel_array.shape[0]):
            frame = pixel_array[i]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(frame)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

            results = self.classification_model.predict(rgb, imgsz=512, show_labels=False, show_boxes=False)
            conf = results[0].probs.top1conf.item()

            if int(results[0].probs.top1) == 0 and conf > best_conf:
                best_frame = enhanced
                best_conf = conf

        return best_frame

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        results = self.segmentation_model.predict(rgb, imgsz=512, conf=0.1, show_labels=False, show_boxes=False)
        return self.draw_segment_contours(frame, results)

    def draw_segment_contours(self, gray_img: np.ndarray, results) -> np.ndarray:
        img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]

        if results[0].masks is None:
            return img

        drawn = set()
        for idx, box in enumerate(results[0].boxes.xywhn):
            cls_id = int(results[0].boxes.cls[idx].item())
            poly = np.asarray(results[0].masks.xyn[idx], dtype=np.float32) * [w, h]

            if cls_id == 0 and 0 not in drawn:
                cv2.polylines(img, [poly.astype(np.int32)], True, (255, 0, 0), 1)
                drawn.add(0)
            elif cls_id == 1 and 1 not in drawn:
                cv2.polylines(img, [poly.astype(np.int32)], True, (0, 255, 0), 1)
                drawn.add(1)
            elif cls_id == 2 and 2 not in drawn:
                cv2.polylines(img, [poly.astype(np.int32)], True, (0, 0, 255), 1)
                drawn.add(2)

        return img

    def detect_lesions(self, image: np.ndarray, output_dir: str) -> list:
        results = self.lesion_detection_model.predict(image, imgsz=512, conf=0.01, show_labels=True)
        output_path = os.path.join(output_dir, "lesion_detection_result.png")
        cv2.imwrite(output_path, results[0].plot(labels=True))

        labels = []
        for r in results:
            for cls_id in r.boxes.cls:
                labels.append(self.lesion_detection_model.names[int(cls_id)])

        return labels

    def update_excel(self, excel_path: str, detections: list):
        df = pd.read_excel(excel_path)
        last_row = df.index[-1]
        target_cols = df.columns[-9:]

        for col in target_cols:
            df.at[last_row, col] = 1 if col in detections else 0

        df.to_excel(excel_path, index=False)

    def prepare_data_for_regression(self, excel_file):
        data = pd.read_excel(excel_file)

        categorical_features = ['Vessel', 'PCI_LAD', 'PCI_LCX', 'PCI_RCA', 'PA_SEX', 'YJ_SEX', 'PA_PRE_PCI',
                                'PA_PRE_CABG',
                                'PA_PRE_MI', 'PA_MI_LOC', 'PA_PRE_CHF', 'PA_CVA', 'PA_DM', 'PA_HBP', 'PA_PRE_CRF',
                                'PA_PRE_CRF_DIA', 'PA_PRE_CRF_DIA_TYPE', 'PA_SMOKING', 'YJ_SMOKING',
                                'YJ_Current_SMOKING',
                                'PA_DYSLIPID', 'PA_FHX_CAD', 'PA_DX', 'YJ_DX', 'YJ_DX_Final', 'LAD_proximal',
                                'LAD_middle',
                                'LAD_distal', 'LCX_proximal', 'LCX_middle', 'LCX_distal', 'RCA_proximal', 'RCA_middle',
                                'RCA_distal']
        continuous_features = ['PA_AGE', 'PA_HEIGHT', 'PA_WEIGHT', 'YJ_BMI', 'PA_PACK_YEAR', 'PA_QUIT_YEAR', 'WBC',
                               'Hb',
                               'Platelet', 'BUN', 'Cr', 'AST', 'ALT', 'LDL-Choleterol', 'HDL-Cholesterol', 'TG']

        expected_columns = categorical_features + continuous_features
        for col in expected_columns:
            if col not in data.columns:
                raise ValueError(f"Missing column in data: {col}")

        return data[expected_columns]

    def make_prediction(self, prepared_data):
        return self.regression_model.predict(prepared_data)

    def process_dicom_and_predict(self, dicom_file, excel_file, output_path):
        import time
        start_time = time.time()

        # Step 1: Trích xuất frame tốt nhất
        best_frame = self.extract_best_frame(dicom_file)
        if best_frame is None:
            raise RuntimeError("No suitable high-quality frame found in DICOM.")

        height, width = best_frame.shape[:2]

        # Step 2: Segment frame tốt nhất
        segmented_frame = self.segment_frame(best_frame)
        segmented_path = os.path.join(output_path, "best_segmented_frame.png")
        cv2.imwrite(segmented_path, segmented_frame)

        # Step 3: Phát hiện tổn thương và overlay kết quả
        lesion_results = self.detect_lesions(segmented_frame, output_path)

        # Step 4: Cập nhật kết quả vào file Excel
        self.update_excel(excel_file, lesion_results)

        # Step 5: Chuẩn bị dữ liệu và dự đoán FFR
        prepared_data = self.prepare_data_for_regression(excel_file)
        prediction = self.make_prediction(prepared_data)

        # Step 6: Tạo summary đẹp như mẫu
        detected_counter = Counter(lesion_results)
        all_classes = list(self.lesion_detection_model.names.values())

        summary_lines = []
        for cls_name in all_classes:
            count = detected_counter.get(cls_name, 0)
            if count > 0:
                summary_lines.append(f"✓ Detected: {count} {cls_name}")
            else:
                summary_lines.append(f"✓ No {cls_name} detected")

        elapsed = time.time() - start_time
        summary_lines.append(f"✓ Processing time: {elapsed:.2f}s")
        summary_text = "\n".join(summary_lines)

        return {
            "lesions": lesion_results,
            "prediction": prediction.tolist(),
            "image_size": {"width": width, "height": height},
            "class_distribution": dict(Counter(lesion_results)),
            "processing_time": f"{elapsed:.2f}s",
            "vessel_type": "eel"
        }
