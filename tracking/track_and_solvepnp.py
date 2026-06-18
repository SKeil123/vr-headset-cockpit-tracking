import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Helpers
def load_intrinsics(npz_path: Path):
    data = np.load(str(npz_path))
    possible_K = ["camera_matrix", "K", "camersMatrix", "mtx"]
    possible_dist = ["dist_coeffs", "dist",  "distCoeffs", "distortion"]

    K = None
    dist = None
    for k in possible_K:
        if k in data:
            K = data[k]
            break
    for k in possible_dist:
        if k in data:
            dist = data[k]
            break

    if K is None:
        raise KeyError(f"Could not find camera matrix in {npz_path}. Keys: {list(data.keys())}")
    if dist is None:
        # If not present, assume zero distortion (not ideal, but keeps pipeline running)
        dist = np.zeros((1, 5), dtype=np.float64)
    
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    return K, dist


def load_points_3d(json_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Returns (ids, pts3d Nx3 float32) in the same order as stored in JSON.
    """
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    pts = obj["points"]
    ids = [p["id"] for p in pts]
    pts3d = np.array([p["xyz"] for p in pts], dtype=np.float32)
    return ids, pts3d

def load_points_pixels(json_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Retruns (ids, pts2d Nx2 float32) in the same order as stored in JSON.
    """
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    pts = obj["points"]
    ids = [p["id"] for p in pts]
    pts2d = np.array([p["uv"] for p in pts], dtype=np.float32)
    return ids, pts2d

def list_frames(frames_dir: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    files = [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No frames found in: {frames_dir}")
    return files

def reprojection_error(pts3d: np.ndarray, pts2d: np.ndarray, rvec, tvec, K, dist) -> float:
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - pts2d, axis=1)
    return float(err.mean())

def draw_debug(img, pts2d, ids, rvec, tvec, pts3d, K, dist):
    out = img.copy()
    # measured points (green)
    for (u, v), pid in zip(pts2d, ids):
        cv2.circle(out, (int(u), int(v)), 4, (0, 255, 0), -1)
        cv2.putText(out, pid, (int(u) + 6, int(v) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    # reprojected points (red)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    for (u, v) in proj:
        cv2.circle(out, (int(u), int(v)), 4, (0, 0, 255), -1)
    return out

# Main pipeline
def main():
    ap = argparse.ArgumentParser(description="Track cockpit points over frames + solvePnP per frame.")
    ap.add_argument("--project-root", type=str, required=True,
                    help=r"Root folder, e.g. C:\Users\shko0\Bachelorarbeit_CameraTracking")
    ap.add_argument("--video-id", type=str, default="video_02",
                    help="frames subfolder name (e.g., video_02)")
    ap.add_argument("--intrinsics", type=str, default=r"camera_calibration\calib_out03\intrinsics.npz",
                    help="Path relative to project root")
    ap.add_argument("--points3d", type=str, default=r"cockpit_videos\cockpit_geometry\cockpit_points_3d.json",
                    help="Path relative to project root")
    ap.add_argument("--points2d0", type=str, default=r"cockpit_videos\cockpit_geometry\cockpit_points_pixels.json",
                    help="Initial 2D points (reference frame) relative to project root")
    ap.add_argument("--frames-dir", type=str, default=r"cockpit_videos\frames",
                    help="Frames directory relative to project root")
    ap.add_argument("--out-dir", type=str, default=r"results\video_pose",
                    help="Output directory relative to project root")
    ap.add_argument("--max-frames", type=int, default=200, help="Limit number of frames (debug)")
    ap.add_argument("--debug-every", type=int, default=20, help="Save debug image evey N frames")
    ap.add_argument("--min-tracked", type=int, default=7,
                    help="Minimum number of valid tracked points required for solvePnP")
    args = ap.parse_args()

    root = Path(args.project_root)
    intr_path = root / args.intrinsics
    pts3d_path = root / args.points3d
    pts2d0_path = root / args.points2d0
    frames_dir = root / args.frames_dir / args.video_id
    out_dir = root / args.out_dir / args.video_id
    dbg_dir = out_dir / "debug_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] project_root:", root)
    print("[INFO] intrinsics:", intr_path)
    print("[INFO] points3d:", pts3d_path)
    print("[INFO] points2d0:", pts2d0_path)
    print("[INFO] frames_dir:", frames_dir)

    K, dist = load_intrinsics(intr_path)
    ids3d, pts3d = load_points_3d(pts3d_path)
    ids2d0, pts2d0 = load_points_pixels(pts2d0_path)

    if ids3d != ids2d0:
        raise ValueError("Point ID order mismatch between 3D and 2D json files.",
                         "They must contain the same IDs in the same order.")
    
    frame_paths = list_frames(frames_dir)[: args.max_frames]
    print(f"[INFO] frames: {len(frame_paths)}")

    # Read first frame
    img0 = cv2.imread(str(frame_paths[0]))
    if img0 is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # --- Scale intrinsics to match video frame size ---
    h0, w0 = img0.shape[:2]   # for 1920x1080 -> w0=1920, h0=1080

    npz = np.load(str(intr_path))
    if "image_size" in npz:
        calib_h, calib_w = map(int, npz["image_size"])     # my calib: [3024, 4032]
        if (calib_w, calib_h) != (w0, h0):
            sx = w0 / calib_w
            sy = h0 / calib_h

            K_scaled = K.copy()
            K_scaled[0, 0] *= sx
            K_scaled[1, 1] *= sy
            K_scaled[0, 2] *= sx
            K_scaled[1, 2] *= sy

            print(f"[INFO] Scaling intrinsics: calib=({calib_w}x{calib_h}) -> frame=({w0}x{h0}) sx={sx:.6f} sy={sy:.6f}")
            K = K_scaled

    # Initialize tracking points
    p0 = pts2d0.reshape(-1, 1, 2).astype(np.float32)

    # Optical flow parameters (KLT)
    lk_params = dict(
        winSize=(41, 41),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # CSV output
    csv_path = out_dir / "poses.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow([
            "frame_idx", "frame_name",
            "n_tracked", "n_inliers",
            "rvec_x", "rvec_y", "rvec_z",
            "tvec_x", "tvec_y", "tvec_z",
            "reproj_err_px",
            "status"
        ])

        prev_gray = gray0
        prev_pts = p0

        # For solvePnP initial guess
        rvec_prev = None
        tvec_prev = None

        for i, fp in enumerate(frame_paths):
            img = cv2.imread(str(fp))
            if img is None:
                print("[WARN] Could not read:", fp)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if i == 0:
                # For first frame, we already have points
                next_pts = prev_pts
                st = np.ones((prev_pts.shape[0], 1), dtype=np.uint8)
            else:
                next_pts, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None, **lk_params
                )
                # Track back from current frame to previous frame
                p_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(
                    gray, prev_gray, next_pts, None, **lk_params
                )

                st_fwd = st.reshape(-1) == 1
                st_bwd = st_back.reshape(-1) == 1

                fb_err = np.linalg.norm(
                    prev_pts.reshape(-1, 2) - p_back.reshape(-1, 2),
                    axis=1
                )

                good_fb = fb_err < 3.0
                good = st_fwd & st_bwd & good_fb
                st = good.astype(np.uint8)

            st = st.reshape(-1)
            tracked_idx = np.where(st == 1)[0]
            n_tracked = int(len(tracked_idx))

            if n_tracked < args.min_tracked:
                print(f"[WARN] Frame {i:04d}: too few tracked points ({n_tracked}), skipping PnP")

                w.writerow([
                    i, fp.name,
                    n_tracked, 0,
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    np.nan,
                    "TOO_FEW_POINTS"
                ])

                prev_gray = gray
                prev_pts = next_pts
                continue

            pts2d = next_pts[tracked_idx].astype(np.float32).reshape(-1, 1, 2)
            pts3d_sel = pts3d[tracked_idx].astype(np.float32).reshape(-1, 1, 3)
            ids_sel = [ids3d[k] for k in tracked_idx.tolist()]

            # SolvePnP (ITERATIVE is stable)
            #use_guess = (rvec_prev is not None and tvec_prev is not None)

            ok_r, rvec_r, tvec_r, inliers = cv2.solvePnPRansac(
                pts3d_sel, pts2d, K, dist,
                iterationsCount=200,
                reprojectionError=12.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP
            )

            n_inl = 0 if (inliers is None) else int(len(inliers))

            ok = False
            if ok_r and n_inl >= 6:
                inl = inliers[:, 0].astype(int)
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d_sel[inl], pts2d[inl],
                    K, dist,
                    rvec=rvec_r, tvec=tvec_r,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                use_guess = (rvec_prev is not None and tvec_prev is not None)
                if use_guess:
                    ok, rvec, tvec = cv2.solvePnP(
                        pts3d_sel, pts2d, K, dist,
                        rvec=rvec_prev, tvec=tvec_prev,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                else:
                    ok, rvec, tvec = cv2.solvePnP(
                        pts3d_sel, pts2d, K, dist,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

            if not ok:
                print(f"[WARN] Frame {i:04d}: solvePnP failed (tracked={n_tracked}, inliers={n_inl})")

                w.writerow([
                    i, fp.name,
                    n_tracked, n_inl,
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    np.nan,
                    "PNP_FAILED"
                ])

                prev_gray = gray
                prev_pts = next_pts
                continue
            
            # 1) RANSAC-Initialschätzung
            """ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d_sel, pts2d, K, dist,
                rvec=rvec_prev if use_guess else None,
                tvec=tvec_prev if use_guess else None,
                useExtrinsicGuess=use_guess,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,
                iterationsCount=200,
                confidence=0.99
            )

            if (not ok) or (inliers is None) or (len(inliers) < args.min_tracked):
                n_inl = 0 if inliers is None else len(inliers)
                print(f"[WARN] Frame {i:04d}: PnPRansac failed / too few inliers ({n_inl})")
                prev_gray = gray
                prev_pts = next_pts
                continue

            # 2) Optional: Refinement mit nur Inliers (stabiliseirt rvec/tvec)
            inl = inliers.reshape(-1)
            pts3d_inl = pts3d_sel[inl]
            pts2d_inl = pts2d[inl]

            ok2, rvec, tvec = cv2.solvePnP(
                pts3d_inl, pts2d_inl, K, dist,
                rvec=rvec, tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not ok2:
                print(f"[WARN] Frame {i:04d}: refine solvePnP failed (keeping RANSAC pose)")"""


            """if use_guess:
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d_sel, pts2d, K, dist,
                    rvec=rvec_prev, tvec=tvec_prev,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d_sel, pts2d, K, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            if not ok:
                print(f"[WARN] Frame {i:04d}: solvePnP failed")
                prev_gray = gray
                prev_pts = next_pts
                continue"""

            # Reprojection error (mean pixel error)
            err_px = reprojection_error(pts3d_sel.reshape(-1,3), 
                                        pts2d.reshape(-1,2), 
                                        rvec, tvec, K, dist)

            # Save csv row
            w.writerow([
                i, fp.name,
                n_tracked, n_inl,
                float(rvec[0,0]), float(rvec[1,0]), float(rvec[2,0]),
                float(tvec[0,0]), float(tvec[1,0]), float(tvec[2,0]),
                float(err_px),
                "OK"
            ])

            # Save debug image occasionally
            if args.debug_every > 0 and (i % args.debug_every == 0):
                dbg = draw_debug(img, 
                                 pts2d.reshape(-1,2), 
                                 ids_sel, 
                                 rvec, tvec, 
                                 pts3d_sel.reshape(-1,3), 
                                 K, dist
                                 )
                cv2.imwrite(str(dbg_dir / f"dbg_{i:04d}.png"), dbg)

            # Update for next iteration
            rvec_prev = rvec
            tvec_prev = tvec
            prev_gray = gray
            prev_pts = next_pts

            if i % 25 == 0:
                print(f"[INFO] Frame {i:04d}: tracked={n_tracked}, reproj={err_px:.2f}px")
        
    print("✅ Done")
    print("CSV:", csv_path)
    print("Debug images:", dbg_dir)

if __name__ == "__main__":
    main()

