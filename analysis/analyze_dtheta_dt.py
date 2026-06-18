import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Compute Δθ/Δt (and optional translation) from pose CSV.")
    p.add_argument("--csv", required=True, type=str, help="Input CSV (e.g., poses_smoothed_ok.csv or poses.csv)")
    p.add_argument("--out-dir", required=True, type=str, help="Output directory for results")
    p.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")
    p.add_argument("--deg", action="store_true", help="Output angles in degrees (default: True)")
    p.add_argument("--omega-thresh", type=float, default=200.0, help="Outlier threshold for ω in deg/s")
    p.add_argument("--dtheta-thresh", type=float, default=10.0, help="Outlier threshold for Δθ in deg")
    return p.parse_args()

def clamp(x, lo=-1.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)

def quat_to_R(qw, qx, qy, qz):
    # Normalize
    q = np.array([qw, qx, qy, qz], dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return None
    qw, qx, qy, qz = q / n

    # Rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=float)
    return R

def rvec_to_R(rv):
    # Rodrigues without OpenCV: use exponential map for small set
    # For stability & simplicity, approximate using axis-angle -> R
    # rv is 3-vector: axis * angle (rad)
    theta = np.linalg.norm(rv)
    if theta == 0:
        return np.eye(3)
    k = rv / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ], dtype=float)
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
    return R

def rel_angle_from_R(R1, R2):
    # R_rel = R1^T R2
    Rrel = R1.T @ R2
    tr = np.trace(Rrel)
    c = clamp((tr - 1.0) / 2.0)
    ang = np.arccos(c)  # rad
    return float(ang)

def pick_rotation(df):
    # Prefer quaternion if present, else rvec, else rvec_sm if present.
    cols = set(df.columns)

    quat_cols = {"qw", "qx", "qy", "qz"}
    if quat_cols.issubset(cols):
        def get_R(i):
            R = quat_to_R(df.at[i, "qw"], df.at[i, "qx"], df.at[i, "qy"], df.at[i, "qz"])
            return R
        return get_R, "quat(qw, qx, qy, qz)"
    
    rvec_sm_cols = {"rvec_x_sm", "rvec_y_sm", "rvec_z_sm"}
    if rvec_sm_cols.issubset(cols):
        def get_R(i):
            rv = np.array([df.at[i, "rvec_x_sm"], df.at[i, "rvec_y_sm"], df.at[i, "rvec_z_sm"]], dtype=float)
            return rvec_to_R(rv)
        return get_R, "rvec_sm"
    
    rvec_cols = {"rvec_x", "rvec_y", "rvec_z"}
    if rvec_cols.issubset(cols):
        def get_R(i):
            rv = np.array([df.at[i, "rvec_x"], df.at[i, "rvec_y"], df.at[i, "rvec_z"]], dtype=float)
            return rvec_to_R(rv)
        return get_R, "rvec"
    
    raise RuntimeError("No usable rotation columns found (need qw, qx, qy, qz OR rvec_* OR rvec_*_sm).")

def pick_translation(df):
    cols = set(df.columns)
    t_sm = {"tvec_x_sm", "tvec_y_sm", "tvec_z_sm"}
    if t_sm.issubset(cols):
        def get_t(i):
            return np.array([df.at[i, "tvec_x_sm"], df.at[i, "tvec_y_sm"], df.at[i, "tvec_z_sm"]], dtype=float)
        return get_t, "tvec_sm"
    
    t = {"tvec_x", "tvec_y", "tvec_z"}
    if t.issubset(cols):
        def get_t(i):
            return np.array([df.at[i, "tvec_x"], df.at[i, "tvec_y"], df.at[i, "tvec_z"]], dtype=float)
        return get_t, "tvec"
    
    return None, None

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # If status exists, keep OK only (for poses.csv)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "OK"].copy()

    # Ensure sorted by frame_idx
    if "frame_idx" in df.columns:
        df = df.sort_values("frame_idx").reset_index(drop=True)

    if len(df) < 2:
        return RuntimeError("Need at least 2 OK frames to compute Δθ/Δt.")
    
    get_R, rot_src = pick_rotation(df)
    get_t, t_src = pick_translation(df)

    rows = []
    for i in range(len(df) - 1):
        f0 = int(df.at[i, "frame_idx"]) if "frame_idx" in df.columns else i
        f1 = int(df.at[i+1, "frame_idx"]) if "frame_idx" in df.columns else i+1
        dt = (f1 - f0) / float(args.fps) if args.fps > 0 else np.nan
        if dt <= 0:
            continue

        R0 = get_R(i)
        R1 = get_R(i+1)
        if R0 is None or R1 is None:
            continue

        dtheta_rad = rel_angle_from_R(R0, R1)
        dtheta_deg = np.degrees(dtheta_rad)
        omega = dtheta_deg / dt

        dtrans = np.nan
        vtrans = np.nan
        if get_t is not None:
            t0 = get_t(i)
            t1 = get_t(i+1)
            dtrans = float(np.linalg.norm(t1 - t0))
            vtrans = dtrans / dt

        rows.append({
            "frame_idx_prev": f0,
            "frame_idx": f1,
            "dt_s": dt,
            "dtheta_deg": dtheta_deg,
            "omega_deg_s": omega,
            "dtrans": dtrans,
            "vtrans": vtrans
        })
    
    out = pd.DataFrame(rows)
    out_csv = out_dir / "dtheta_dt.csv"
    out.to_csv(out_csv, index=False)

    # Summary
    def stats(series):
        s = series.dropna().to_numpy()
        if len(s) == 0:
            return "n=0"
        return (f"n={len(s)}, mean={np.mean(s):.3f}, median={np.median(s):.3f}, "
                f"p95={np.percentile(s,95):.3f}, max={np.max(s):.3f}")
    
    omega_s = out["omega_deg_s"]
    dtheta_s = out["dtheta_deg"]
    dt_s = out["dt_s"]

    outliers = out[(out["omega_deg_s"] > args.omega_thresh) | (out["dtheta_deg"] > args.dtheta_thresh)].copy()

    summary_txt = out_dir / "dtheta_dt_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Input CSV: {args.csv}\n")
        f.write(f"FPS: {args.fps}\n")
        f.write(f"Rotation source: {rot_src}\n")
        f.write(f"Translation source: {t_src}\n\n")
        f.write("Δt statistics:\n")
        f.write(stats(dt_s) + "\n\n")
        f.write("Δθ (deg) statistics:\n")
        f.write(stats(dtheta_s) + "\n\n")
        f.write("ω (deg/s) statistics:\n")
        f.write(stats(omega_s) + "\n\n")
        f.write(f"Outlier rule: ω > {args.omega_thresh} OR Δθ > {args.dtheta_thresh}\n")
        f.write(f"Outlier count: {len(outliers)}\n")
        if len(outliers) > 0:
            f.write("Outlier frame transitions (prev -> curr):\n")
            for _, r in outliers.iterrows():
                f.write(f"- {int(r['frame_idx_prev'])} -> {int(r['frame_idx'])} : "
                        f"dt={r['dt_s']:.4f}s, dθ={r['dtheta_deg']:.2f}deg, ω={r['omega_deg_s']:.2f}deg/s\n")
                
    # Plots
    x = np.arange(len(out))

    plt.figure()
    plt.plot(x, out["dtheta_deg"].to_numpy())
    plt.title("Δθ over time")
    plt.xlabel("transition index")
    plt.ylabel("dtheta_deg")
    plt.tight_layout()
    plt.savefig(out_dir / "dtheta_over_time.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, out["omega_deg_s"].to_numpy())
    plt.title("ω = Δθ/Δt over time")
    plt.xlabel("transition index")
    plt.ylabel("omega_deg_s")
    plt.tight_layout()
    plt.savefig(out_dir / "omega_over_time.png", dpi=200)
    plt.close()

    if get_t is not None:
        plt.figure()
        plt.plot(x, out["dtrans"].to_numpy())
        plt.title("Δtrans over time")
        plt.xlabel("transition index")
        plt.ylabel("dtrans")
        plt.tight_layout()
        plt.savefig(out_dir / "dtrans_over_time.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(x, out["vtrans"].to_numpy())
        plt.title("v = Δtrans/Δt over time")
        plt.xlabel("transition index")
        plt.ylabel("vtrans")
        plt.tight_layout()
        plt.savefig(out_dir / "vtrans_over_time.png", dpi=200)
        plt.close()

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {summary_txt}")
    print(f"[OK] Plots written to: {out_dir}")

if __name__ == "__main__":
    main()