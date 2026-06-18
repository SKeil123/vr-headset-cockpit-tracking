from PIL import Image, ImageOps
from pathlib import Path

# 원본 이미지 폴더
SRC_DIR = Path(r"C:\Users\shko0\Bachelorarbeit_CameraTracking\camera_calibration\calibration_images")

# 출력 폴더
DST_DIR = Path(r"C:\Users\shko0\Bachelorarbeit_CameraTracking\camera_calibration\calibration_images_fixed")

DST_DIR.mkdir(parents=True, exist_ok=True)

count = 0
for img_path in SRC_DIR.glob("*.jpg"):
    img = Image.open(img_path)

    img_fixed = ImageOps.exif_transpose(img)

    out_path = DST_DIR / img_path.name
    img_fixed.save(out_path, quality=95)

    count += 1
    print(f"[OK] fixed: {img_path.name}")

print(f"\nDone. {count} images written to:\n{DST_DIR}")

