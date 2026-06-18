# Kamera Kalibrieren für Handykamera (iPhone 14 Pro)
# Features:
# 1. Robuste Eckenfindung : zuerst findChessboardCornerSB, Fallback auf klassische Methode
# 2. CLI-Parameter : Patterngröße, Quadratgröße, Flags, Ausgabeoptionen
# 3. Optional : Rationales Verzeruungsmodell (k1..k6, besser für eine Handykamera) 
# 4. Optional : Video-Frames sampeln
# 5. Logging : summary.txt, per_image_errors.csv, optionale Eckenvizualisierungen
# 6. Speicheren rvecs/tvecs (Extrinsics je Bild) für nachgelagerte PnP-/Pose-Analysen
# 7. Entzerren(Equalizing) alle Bilder; optional Side-by-Side speichern



import argparse
from pathlib import Path
import sys
import csv
import cv2
import numpy as np
from datetime import datetime
import json

# Argumente parsen
def parse_args():
    p = argparse.ArgumentParser(
        description="Camera calibration using checkerboard images (robust SB detection, optional rational model & video sampling)"
    )

    # Basis
    p.add_argument("--images-dir", type=str, default="calibration_images", help="Folder with checkerboard images")
    p.add_argument("--pattern-cols", type=int, default=10, help="Number of inner corners horizontally")
    p.add_argument("--pattern-rows", type=int, default=7, help="Number of inner corners vertically")
    p.add_argument("--square-mm", type=float, default=15.0, help="Size of one square in millimeters")

    # Ausgabe & Visualisierung
    p.add_argument("--output-root", type=str, default="results", help="Output root directory")
    p.add_argument("--draw-detections", action="store_true", help="Draw detected corners and save them to logs/detections")
    p.add_argument("--save-side-by-side", action="store_true", help="Save side-by-side comparison for undistortion")
    p.add_argument("--ext", type=str, default="jpg")

    # Flags & Modelle
    p.add_argument("--force-aspect-fixed", action="store_true", help="Use CALIB_FIX_ASPECT_RATIO flag")
    p.add_argument("--zero-tangent", action="store_true", help="Force tangential distortion (p1, p2) to zero")   # tangent왜곡을 0으로 고정, 실제 왜곡이 있는데 0으로 고정하면 오차 up
    p.add_argument("--no-k3", action="store_true", help="Disable k3 coefficient (CALIB_FIX_K3)")
    p.add_argument("--rational", action="store_true", help="Use CALIB_RATIONAL_MODEL")

    # Video-Unterstützung
    p.add_argument("--video", type=str, default="", help="Optional video file to sample frames from")
    p.add_argument("--video-step", type=int, default=3, help="Use every N-th frame from the video")

    return p.parse_args()

# Hilfsfunktionen
def ensure_dirs(root: Path):
    """Stellen sicher, dass Ausgabeverzeichnisse existieren."""
    (root / "undistorted").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

def find_images(images_dir: Path, exts):
    """Finden Bilddateien mit gegebenen Erweiterungen (groß/klein)."""
    imgs = set()
    for ext in exts:
        imgs.update(images_dir.glob(f"*.{ext}"))
        #imgs.extend(sorted(images_dir.glob(f"*.{ext}")))
        #imgs.extend(sorted(images_dir.glob(f"*.{ext.upper()}")))
    return sorted(imgs)

def try_find_corners(gray, pattern_size):
    """Versuchen zuerst SB(robust), dann klassische Eckenerkennung inkl. Subpixeloptimierung."""
    try:
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if ok:
            return True, corners.astype(np.float32)
    except Exception:
        pass
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 12-6)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return True, corners

def build_object_points(pattern_size, square_mm):
    """Erstellen 3D Punkte des Checkerboards im Weltkoordinatensystem (z = 0)."""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= (square_mm / 1000.0)  # mm -> m
    return objp

def save_intrinsics_npz(out_path: Path, **kwargs):
    """Speichern Kalibrierung als NPZ (leicht weiterzuverwenden)."""
    np.savez(out_path, **kwargs)

def undistort_image(img, K, dist, keep_size=True):
    """Erzeugen entzerrte Version mit optimaler Kameramatrix."""
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    if keep_size:
        dst = cv2.undistort(img, K, dist, None, newK)
    else:
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16FC2)
        dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return dst


# Hauptablauf
def main():
    args = parse_args()
    root = Path(args.output_root)
    ensure_dirs(root)
    logs_dir = root / "logs"
    det_viz_dir = logs_dir / "detections" if args.draw_detections else None
    if det_viz_dir is not None:
        det_viz_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path(args.images_dir)
    exts = [e.strip() for e in args.ext.split(",") if e.strip()]
    exts = list(set(exts))   # Verdopplung löschen 
    pattern_size = (args.pattern_cols, args.pattern_rows)
    objp = build_object_points(pattern_size, args.square_mm)

    objpoints = []
    imgpoints = []
    used_files = []
    img_files = find_images(images_dir, exts)
    print("\n[DEBUG] Found image files:")
    for f in img_files:
        print("   ", f)
    print("[DEBUG] Total files found:", len(img_files), "\n")

    if len(img_files) == 0:
        print(f"[WARN] No images found in: {images_dir.resolve()} (*.{args.ext})")
    image_size = None

    print(f"[INFO] Processing images in '{images_dir}'...")
    for f in img_files:
        arr = np.fromfile(str(f), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = try_find_corners(gray, pattern_size)
        if not ok:
            print(f"[WARN] Corners not found -> skippend: {f.name}")
            continue
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used_files.append(f.name)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        if det_viz_dir is not None:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
            cv2.imwrite(str(det_viz_dir / f"{f.stem}_corners.jpg"), vis)

    if len(objpoints) < 8:
        print(f"[ERROR] Only {len(objpoints)} valid frames/images. At least 8 required.", file=sys.stderr)
        sys.exit(2)

    # Kalibreirungsflags
    flags = 0
    if args.force_aspect_fixed:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if args.zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if args.no_k3:
        flags |= cv2.CALIB_FIX_K3
    if args.rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    
    print("[FLAGS]", end=" ")
    if flags == 0:
        print("No flags set.")
    else:
        if flags & cv2.CALIB_FIX_K3:
            print("CALIB_FIX_K3", end=" ")
        if flags & cv2.CALIB_RATIONAL_MODEL:
            print("CALIB_RATIONAL_MODEL", end=" ")
        if flags & cv2.CALIB_ZERO_TANGENT_DIST:
            print("CALIB_ZERO_TANGENT_DIST", end=" ")
        if flags & cv2.CALIB_FIX_ASPECT_RATIO:
            print("CALIB_FIX_ASPECT_RATIO", end=" ")
        print()
    
    # Kalibrierung
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=flags, criteria=criteria)

    print(f"[RESULT] RMS reprojection error: {rms:.6f}")

    per_image_errors = []
    for i, (op, ip, rv, tv) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        proj = proj.reshape(-1, 2)
        err = np.sqrt(np.mean(np.sum((ip.reshape(-1, 2) - proj)**2, axis=1)))
        per_image_errors.append(float(err))

    intrinsics_path = root / "intrinsics.npz"
    save_intrinsics_npz(
        intrinsics_path,
        camera_matrix = K,
        dist_coeffs = dist,
        rvecs = np.array(rvecs, dtype=object),
        tvecs = np.array(tvecs, dtype=object),
        image_size = np.array(image_size),
        pattern_cols = args.pattern_cols,
        pattern_rows = args.pattern_rows,
        square_mm = args.square_mm,
        rms = rms,
        per_image_errors = np.array(per_image_errors),
        used_files = np.array(used_files, dtype=object),
        flags = int(flags),
        created_at = str(datetime.now()),
        opencv_version = cv2.__version__)
    
    print(f"[SAVE] Intrinsics saved: {intrinsics_path}")

    # Blender Calibration
    blender_json = root / "logs" / "calib_for_blender.json"
    frames = []
    for fname, rv, tv in zip(used_files, rvecs, tvecs):
        frames.append({
            "filename" : fname,
            "rvec": rv.reshape(-1).tolist(),
            "tvec": tv.reshape(-1).tolist()
        })

    calib_data = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "pattern_cols": int(args.pattern_cols),
        "pattern_rows": int(args.pattern_rows),
        "square_mm" : float(args.square_mm),
        "rms": float(rms),
        "frames": frames
    }

    with open(blender_json, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    print(f"[SAVE] Blender JSON saved: {blender_json}")


    #neu
    extr_csv = root / "logs" / "extrinsics.csv"
    with open(extr_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["index", "filename", 
                    "rvec_x", "rvec_y", "rvec_z",
                    "tvec_x", "tvec_y", "tvec_z"])
        for i, (fname, rv, tv) in enumerate(zip(used_files, rvecs, tvecs)):
            r = rv.reshape(-1)
            t = tv.reshape(-1)
            w.writerow([i, fname, r[0], r[1], r[2], t[0], t[1], t[2]])
    
    extr_dir = root / "extrinsics"
    extr_dir.mkdir(exist_ok=True)
    for i, (fname, rv, tv) in enumerate(zip(used_files, rvecs, tvecs)):
        out_file = extr_dir / f"{fname}_extrinsics.txt"
        with open(out_file, "w", encoding="utf-8") as fw:
            fw.write(f"Imgae: {fname}\n")
            fw.write(f"rvec: {rv.reshape(-1)}\n")
            fw.write(f"tvec: {tv.reshape(-1)}\n")


    summary_txt = (root / "logs" / "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as fw:
        fw.write("# Camera Calibraiton Summary\n")
        fw.write(f"opencv_version: {cv2.__version__}\n")
        fw.write(f"image_size (w, h): {image_size}\n")
        fw.write(f"pattern (cols, rows): ({args.pattern_cols}, {args.pattern_rows})\n")
        fw.write(f"square_mm: {args.square_mm}\n")
        fw.write(f"flags: {flags}\n")
        fw.write(f"RMS: {rms:.6f}\n\n")
        fw.write("camera_matrix (K):\n")
        fw.write(np.array2string(K, precision=6))
        fw.write("\n\ndist_coeffs:\n")
        fw.write(np.array2string(dist, precision=6))
        fw.write("\n")

    print(f"[SAVE] Summary file saved: {summary_txt}")

    errors_csv = (root / "logs" / "per_image_errors.csv")
    with open(errors_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["index", "filename", "reproj_error(px)", "num_points"])
        for i, (fname, err) in enumerate(zip(used_files, per_image_errors)):
            w.writerow([i, fname, f"{err:.6f}", len(imgpoints[i])])

    undist_dir = root / "undistorted"
    for f in img_files:
        arr = np.fromfile(str(f), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        dst = undistort_image(img, K, dist)
        if args.save_side_by_side:
            h = max(img.shape[0], dst.shape[0])
            w = img.shape[1] + dst.shape[1]
            side = np.zeros((h, w, 3), dtype=img.dtype)
            side[:img.shape[0], :img.shape[1]] = img
            side[:dst.shape[0], img.shape[1]:img.shape[1] + dst.shape[1]] = dst
            out_path = undist_dir / f"{f.stem}_side_by_side.jpg"
            cv2.imwrite(str(out_path), side)
        else:
            out_path = undist_dir / f.name
            cv2.imwrite(str(out_path), dst)

    print(f"[DONE] Used {len(objpoints)} valid frame(s). Results saved in '{root.resolve()}'.")

if __name__ == "__main__":
    main()


