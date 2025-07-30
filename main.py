from concurrent.futures import ThreadPoolExecutor
import io
import os
import time
from uuid import uuid4
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests 
from model.detection_eel import YoloDetection_EEL
from model.segment_calcium import YoloSegmentor_Calcium
from model.segmention import YoloSegmentor
from model.detection import YoloDetection
from model.angioFFR import AngioFFR
from PIL import Image, ImageSequence


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(os.path.join(STATIC_DIR, "results"), exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://oct-demo.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def download_from_gdrive(file_id: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"Model already exists: {dest_path}")
        return
    print(f"Downloading model to {dest_path}...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print("✅ Download complete.")

def download_all_weights():
    os.makedirs("weights", exist_ok=True)

    download_from_gdrive(
        "1CEtx836c2tUbCCpdZHrnq5jonaG1yr-1",
        "weights/241003_yolov8x_lca_dropout05_best.pt"
    )
    download_from_gdrive(
        "1w-QQF2R_YJppd0nwFSbz_O3rqnjr7rGq",
        "weights/241003_yolov8x_rca_dropout05_best.pt"
    )
    download_from_gdrive(
        "1ONgna1eyvSTLgiXp3ULLwmo0anoNwBnf",
        "weights/20250430_segment_200epoch_yolov8n_best.pt"
    )
    download_from_gdrive(
        "1S1M0BHn9V-p16F3jyAfPuR4_OjEeWIln",
        "weights/cls_400epoch_50patience_best.pt"
    )
    download_from_gdrive(
        "1fKW0nslRwnadJ5mM8N1sQGwyjjtscY44",
        "weights/train_250414_yolov8n_300epoch_imgsz1024.pt"
    )
    download_from_gdrive(
        "1W-iF-4qMFGPXOAF2FvvBZZ1OATPNepg2",
        "weights/train_250524_yolov8m_500epoch_imgsz1024.pt"
    )
    download_from_gdrive(
        "1DSqJUfaH9WBwEudZ7kQZimSaBduh_i9w",
        "weights/240403_calcium_2022_30epoch_best.pt"
    )
    download_from_gdrive(
        "1wkyFYa_jYd0doyFpID1Loewar93mdcbU",
        "weights/eel_2093img_100epoch.pt"
    )
    download_from_gdrive(
        "1UEn_ILA6N_RlAl5LmmzYBQNO56_mC5Vr",
        "weights/train_240611_eel_keypoint_new_LUT.pt"
    )

download_all_weights()
detection = YoloDetection()
segmentor = YoloSegmentor()
angioffr = AngioFFR()
# === API ===
    
@app.post("/uploadImage")
async def upload_image(file: UploadFile):
    temp_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1]
    tif_path = f"/tmp/{temp_id}{ext}"

    with open(tif_path, "wb") as f:
        f.write(await file.read())

    im = Image.open(tif_path)
    frames = []
    i = 0
    while True:
        try:
            im.seek(i)
            frames.append((i, im.copy()))
            i += 1
        except EOFError:
            break

    preview_dir = "static/preview"
    os.makedirs(preview_dir, exist_ok=True)

    def save_frame(frame_info):
        i, frame = frame_info
        out_path = os.path.join(preview_dir, f"{temp_id}_frame_{i}.png")
        frame.convert("RGB").save(out_path)
        return f"/static/preview/{temp_id}_frame_{i}.png"

    with ThreadPoolExecutor(max_workers=8) as executor:
        frame_paths = list(executor.map(save_frame, frames))

    return {
        "image_urls": frame_paths,
        "temp_filename": f"{temp_id}{ext}"
    }

@app.post("/predict")
async def predict_by_model(
    file: UploadFile = File(...),
    model: str = Form(...),
    task: str = Form(None)  # chỉ dùng khi model1
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
    # elif model == "model2":
    #     segmentor = YoloSegmentor_Calcium()
    #     result = segmentor.predict(content, file.filename)

    #     if result["type"] == "tif":
    #         return JSONResponse(content={
    #             "is_tif": True,
    #             "model": model,
    #             "task": "segmentation",
    #             "frames": result["frames"],  # mỗi frame có frame_index, url, summary, boxes,...
    #             "frame_count": result["frame_count"]
    #         })
    #     else:
    #         frame = result["frames"][0]
    #         return JSONResponse(content={
    #             "is_tif": False,
    #             "model": model,
    #             "task": "segmentation",
    #             "image_url": frame["url"],
    #             "image_size": result.get("image_size", {}),
    #             "summary": frame["summary"],
    #             "boxes": frame["boxes"],
    #             "class_distribution": {},
    #             "vessel_type": "calcium"
    #         })
    elif model == "model2":
        if task == "predict":
            segmentor = YoloSegmentor_Calcium()
            result = segmentor.predict(content, file.filename)

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
            detector = YoloDetection_EEL()
            result = detector.predict(content, file.filename)

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

        else:
            return JSONResponse(status_code=400, content={"error": "Missing or invalid task for model2"})

    # ===================== MODEL 3 - Giữ nguyên =====================
    elif model == "model3":
        if ext in [".tif", ".tiff"]:
            result = angioffr.predict_tif_file(content, file.filename, results_dir)
            return JSONResponse(content={
                "is_tif": True,
                "model": model,
                "task": "segmentation",
                "frame_count": result["frame_count"],
                "frames": result["frames"]
            })
        else:
            result = angioffr.predict(content, file.filename)
            output_img_path = os.path.join(BASE_DIR, result["output_path"].lstrip("/"))
            img = cv2.imread(output_img_path)
            height, width = img.shape[:2]

            boxes_info = result.get("boxes_info", [])
            class_count = result.get("class_distribution", {})

            summary = (
                ", ".join(f"{k}: {v}" for k, v in class_count.items())
                if isinstance(class_count, dict) and class_count else "Không có"
            )

            return JSONResponse(content={
                "is_tif": False,
                "model": model,
                "task": "segmentation",
                "image_url": result["output_path"],
                "boxes": boxes_info,
                "summary": summary,
                "class_distribution": class_count,
                "image_size": {
                    "width": width,
                    "height": height
                },
                "vessel_type": result["vessel_type"]
            })

    # ===================== Invalid Model =====================
    else:
        return JSONResponse(status_code=400, content={"error": f"Invalid model: {model}"})
