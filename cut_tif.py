from PIL import Image
import os

def save_first_20_frames_as_tif(input_tif: str, output_tif: str):
    with Image.open(input_tif) as img:
        frames = []
        for i in range(50):
            try:
                img.seek(i)
                frames.append(img.copy())  # copy để tránh lazy loading
            except EOFError:
                print(f"❗ File chỉ có {i} frame, kết thúc sớm.")
                break

        if frames:
            # Lưu lại thành 1 file TIF nhiều trang
            frames[0].save(
                output_tif,
                save_all=True,
                append_images=frames[1:],
                compression="tiff_deflate"
            )
            print(f"✅ Đã lưu {len(frames)} frame vào {output_tif}")
        else:
            print("❌ Không có frame nào được lưu.")


save_first_20_frames_as_tif("/home/lab/son/project/backend/temp/F315_LAD_fu.tif", "cropped_50frames.tif")
