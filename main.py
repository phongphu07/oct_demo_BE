from concurrent.futures import ThreadPoolExecutor
import io
import os
import shutil
import tempfile
import time
from uuid import uuid4
import uuid
import numpy as np
from custom_static import CustomStaticFiles
import cv2
import uuid
import pydicom   
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
from sklearn import pipeline
from model.detection_eel import YoloDetection_EEL
from model.segment_calcium import YoloSegmentor_Calcium
from model.segmention import YoloSegmentor
from model.detection import YoloDetection
from model.angioFFR import  CombinedPipeline
from PIL import Image, ImageSequence
from feedback import router as feedback_router

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(os.path.join(STATIC_DIR, "results"), exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://oct-demo.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


    
app.include_router(feedback_router)
app.mount("/static", CustomStaticFiles(directory=STATIC_DIR), name="static")
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
def download_from_gdrive(file_id: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"Model already exists: {dest_path}")
        return
    print(f"Downloading model to {dest_path}...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")


detection = YoloDetection()
segmentor = YoloSegmentor()
segmentor2 = YoloSegmentor_Calcium()
detector2 = YoloDetection_EEL()
pipeline = CombinedPipeline()

STATIC_RESULT_DIR = os.path.join("output")
EXCEL_TEMPLATE_PATH = os.path.join("example_image", "model3", "FFR_Input_Format.xlsx")
os.makedirs(STATIC_RESULT_DIR, exist_ok=True)
# === API ===

@app.post("/uploadImage")
async def upload_image(file: UploadFile):
    temp_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1].lower()
    raw_path = f"/tmp/{temp_id}{ext}"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    preview_dir = os.path.join("static", "preview")
    os.makedirs(preview_dir, exist_ok=True)

    frames = []

    if ext == ".dcm":
        ds = pydicom.dcmread(raw_path)
        arr = ds.pixel_array 
        
        arr = np.squeeze(arr)

        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                frame = arr[i]
                if frame.dtype != np.uint8:
                    frame = (frame / frame.max() * 255).astype(np.uint8)
                image = Image.fromarray(frame).convert("RGB")
                frames.append((i, image))
        elif arr.ndim == 2:
            if arr.dtype != np.uint8:
                arr = (arr / arr.max() * 255).astype(np.uint8)
            image = Image.fromarray(arr).convert("RGB")
            frames = [(0, image)]
        else:
            raise ValueError(f"Unsupported DICOM shape: {arr.shape}")


    elif ext in [".tif", ".tiff"]:
        im = Image.open(raw_path)
        i = 0
        while True:
            try:
                im.seek(i)
                frames.append((i, im.copy()))
                i += 1
            except EOFError:
                break
    else:
        im = Image.open(raw_path).convert("RGB")
        frames = [(0, im)]

    def save_frame(frame_info):
        i, frame = frame_info
        out_path = os.path.join(preview_dir, f"{temp_id}_frame_{i}.png")
        frame.convert("RGB").save(out_path)
        return f"/static/preview/{temp_id}_frame_{i}.png"

    with ThreadPoolExecutor(max_workers=8) as executor:
        frame_paths = list(executor.map(save_frame, frames))

    try:
        strip_images = [np.array(f[1].convert("RGB")) for f in frames]
        merged_strip = np.hstack(strip_images)
    except Exception as e:
        print("Failed to merge strip:", e)
        merged_strip = np.array(frames[0][1].convert("RGB")) 

    merged_strip_path = os.path.join(preview_dir, f"{temp_id}_strip.png")
    cv2.imwrite(merged_strip_path, cv2.cvtColor(merged_strip, cv2.COLOR_RGB2BGR))

    return {
        "image_urls": frame_paths,
        "unrolled_url": f"/static/preview/{temp_id}_strip.png",
        "temp_filename": f"{temp_id}{ext}"
    }

# @app.post("/predict_angio")
# async def predict_angio(dicom_file: UploadFile = File(...)):
#     try:
#         ext = os.path.splitext(dicom_file.filename)[1].lower()

#         with tempfile.TemporaryDirectory() as tmpdir:
#             file_path = os.path.join(tmpdir, dicom_file.filename)
#             output_dir = os.path.join(tmpdir, "output")
#             os.makedirs(output_dir, exist_ok=True)

#             # Lưu file đầu vào
#             with open(file_path, "wb") as f:
#                 shutil.copyfileobj(dicom_file.file, f)

#             # === DICOM ===
#             if ext == ".dcm":
#                 result = pipeline.predict(dicom_path=file_path, output_dir=output_dir)
#                 image_path = os.path.join(output_dir, "best_segmented_frame.png")
#             else:
#                 # === Ảnh thông thường ===
#                 result = pipeline.predict_image(image_path=file_path, output_dir=output_dir)
#                 image_path = os.path.join(output_dir, "predicted_image.png")

#             # Kiểm tra ảnh đầu ra
#             if not os.path.exists(image_path):
#                 return JSONResponse(status_code=500, content={"error": "Predicted image not found."})

#             # Chuyển vào thư mục static để frontend truy cập
#             static_result_dir = os.path.join(STATIC_DIR, "results")
#             os.makedirs(static_result_dir, exist_ok=True)
#             final_path = os.path.join(static_result_dir, f"{uuid4().hex}.png")
#             shutil.copy(image_path, final_path)

#             img = cv2.imread(final_path)
#             height, width = img.shape[:2]

#             return JSONResponse(content={
#                 "is_tif": ext in [".tif", ".tiff"],
#                 "model": "model3",
#                 "task": "segment",
#                 "image_url": f"/static/results/{os.path.basename(final_path)}",
#                 "image_size": {"width": width, "height": height},
#                 "summary": f"Detected {len(result['lesions'])} lesion(s): " + ", ".join(result['lesions']),
#                 "boxes": result.get("boxes", []),
#                 "class_distribution": result.get("class_distribution", {}),
#                 "vessel_type": result.get("vessel_type", "eel"),
#                 "ffr_prediction": result.get("prediction")
#             })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return JSONResponse(status_code=500, content={"error": str(e)})

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(os.path.join(STATIC_DIR, "results"), exist_ok=True)

@app.post("/predict_angio")
async def predict_angio(dicom_file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(dicom_file.filename)[1].lower()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, dicom_file.filename)
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Lưu file đầu vào
            with open(file_path, "wb") as f:
                shutil.copyfileobj(dicom_file.file, f)

            # === Gọi pipeline ===
            if ext == ".dcm":
                result = pipeline.predict(dicom_path=file_path, excel_path='example_image/model3/FFR_Regression_Test_F213_RCA1.xlsx', output_dir=output_dir)
                segmented_path = os.path.join(output_dir, "best_segmented_frame.png")
                lesion_path = os.path.join(output_dir, "lesion_detection_result.png")
            else:
                result = pipeline.predict_image(image_path=file_path, output_dir=output_dir)
                segmented_path = os.path.join(output_dir, "predicted_image.png")
                lesion_path = segmented_path  # same for regular images

            # Kiểm tra ảnh đầu ra
            if not os.path.exists(lesion_path):
                return JSONResponse(status_code=500, content={"error": "Prediction image not found."})

            # Chuyển ảnh sang thư mục static để frontend đọc
            static_result_dir = os.path.join(STATIC_DIR, "results")
            final_image_name = f"{uuid.uuid4().hex}.png"
            final_path = os.path.join(static_result_dir, final_image_name)
            shutil.copy(lesion_path, final_path)

            # Lấy kích thước ảnh
            img = cv2.imread(final_path)
            height, width = img.shape[:2]

            return JSONResponse(content={
                "is_tif": ext in [".tif", ".tiff"],
                "model": "model3",
                "task": "segment",
                "image_url": f"/static/results/{final_image_name}",
                "image_size": {"width": width, "height": height},
                "summary": f"Detected {len(result['lesions'])} lesion(s): " + ", ".join(result['lesions']),
                "boxes": result.get("boxes", []),  # Ensure this is returned from your pipeline
                "class_distribution": result.get("class_distribution", {}),
                "vessel_type": result.get("vessel_type", "eel"),
                "ffr_prediction": result.get("prediction")
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict")
async def predict_by_model(
    file: UploadFile = File(...),
    model: str = Form(...),
    task: str = Form(None) 
):
    content = await file.read()
    ext = os.path.splitext(file.filename)[1].lower()
    results_dir = os.path.join(STATIC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ===================== MODEL 1 =====================
    if model == "model1":
        if task == "predict":
            if ext in [".tif", ".tiff"]:
                st = time.time()
                result_urls = detection.predict_tif_file(content, file.filename, output_dir=results_dir)
                ed = time.time()
                print(f"predict_tif time: {ed - st} s")
                return JSONResponse(content={
                    "is_tif": True,
                    "model": model,
                    "task": "predict",
                    "frames": result_urls,
                    "frame_count": len(result_urls)
                })
            else:
                output_filename = f"{uuid4().hex}{ext}"
                output_path = os.path.join(results_dir, output_filename)
                result = detection.predict_single_image(content, file.filename, output_path)
                return JSONResponse(content={
                    "is_tif": False,
                    "model": model,
                    "task": "predict",
                    "image_url": f"/static/results/{output_filename}",
                    "boxes": result["boxes"],
                    "image_size": result["image_size"],
                    "summary": result["summary"]
                })

        elif task == "segment":
            result = segmentor.predict(content, file.filename)
            if result["type"] == "tif":
                return JSONResponse(content={
                    "is_tif": True,
                    "model": model,
                    "task": "segment",
                    "frames": result["frames"],
                    "frame_count": result["frame_count"]
                })
            else:
                frame = result["frames"][0]
                return JSONResponse(content={
                    "is_tif": False,
                    "model": model,
                    "task": "segment",
                    "image_url": frame["url"],
                    "boxes": frame["boxes"],
                    "summary": frame["summary"],
                    "image_size": result.get("image_size", {})
                })
        else:
            return JSONResponse(status_code=400, content={"error": "Missing or invalid task for model1"})

    # ===================== MODEL 2 - Segment calcium =====================
    elif model == "model2":
        if task == "predict":
            if segmentor2 is None:
                return JSONResponse(status_code=500, content={"error": "Model segmentor2 not loaded"})
            result = segmentor2.predict(content, file.filename)

            if result["type"] == "tif":
                return JSONResponse(content={
                    "is_tif": True,
                    "model": model,
                    "task": "predict",
                    "frames": result["frames"],
                    "frame_count": result["frame_count"]
                })
            else:
                frame = result["frames"][0]
                return JSONResponse(content={
                    "is_tif": False,
                    "model": model,
                    "task": "predict",
                    "image_url": frame["url"],
                    "image_size": result.get("image_size", {}),
                    "summary": frame["summary"],
                    "boxes": frame["boxes"],
                    "class_distribution": {},
                    "vessel_type": "calcium"
                })

        elif task == "segment":
            if detector2 is None:
                return JSONResponse(status_code=500, content={"error": "Model detector2 not loaded"})
            result = detector2.predict(content, file.filename)

            if result["type"] == "tif":
                return JSONResponse(content={
                    "is_tif": True,
                    "model": model,
                    "task": "segment",
                    "frames": result["frames"],
                    "frame_count": result["frame_count"]
                })
            else:
                frame = result["frames"][0]
                return JSONResponse(content={
                    "is_tif": False,
                    "model": model,
                    "task": "segment",
                    "image_url": frame["url"],
                    "image_size": result.get("image_size", {}),
                    "summary": frame["summary"],
                    "boxes": frame["boxes"],
                    "class_distribution": {},
                    "vessel_type": "eel"
                })
    elif model == "model3":
        try:
            ext = os.path.splitext(file.filename)[1].lower()
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, file.filename)
                output_dir = os.path.join(tmpdir, "output")
                os.makedirs(output_dir, exist_ok=True)

                # Lưu file đầu vào
                with open(file_path, "wb") as f:
                    f.write(content)

                if ext == ".dcm":
                    # Gọi đúng hàm 3 tham số
                    result = pipeline.process_dicom_and_predict(
                        dicom_file=file_path,
                        excel_file=EXCEL_TEMPLATE_PATH,
                        output_path=output_dir
                    )
                    image_path = os.path.join(output_dir, "lesion_detection_result.png")
                else:
                    result = pipeline.predict_image(image_path=file_path, output_dir=output_dir)
                    image_path = os.path.join(output_dir, "predicted_image.png")

                if not os.path.exists(image_path):
                    return JSONResponse(status_code=500, content={"error": "Predicted image not found."})

                static_result_dir = os.path.join(STATIC_DIR, "results")
                os.makedirs(static_result_dir, exist_ok=True)
                final_path = os.path.join(static_result_dir, f"{uuid4().hex}.png")
                shutil.copy(image_path, final_path)

                img = cv2.imread(final_path)
                height, width = img.shape[:2]

                return JSONResponse(content={
                    "is_tif": ext in [".tif", ".tiff"],
                    "model": "model3",
                    "task": "segment",
                    "image_url": f"/static/results/{os.path.basename(final_path)}",
                    "image_size": {"width": width, "height": height},
                    "summary": f"Detected classes: {', '.join([f'{k}: {v}' for k, v in result.get('class_distribution', {}).items()])}\nProcessing time: {result.get('processing_time', '')}",
                    "boxes": result.get("boxes", []),
                    "class_distribution": result.get("class_distribution", {}),
                    "vessel_type": result.get("vessel_type", "eel"),
                    "ffr_prediction": result.get("prediction")
                })


        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})


        else:
            return JSONResponse(status_code=400, content={"error": "Missing or invalid task for model2"})

