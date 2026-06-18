import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# Rotation helpers (rvec <-> quat)
def rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    """rvec shape (3,) -> quat (w,x,y,z)"""
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return rotmat_to_quat(R)

def quat_to_rvec(q: np.ndarray) -> np.ndarray:
    """quat (w,x,y,z) -> rvec shape (3,)"""
    R = quat_to_rotmat(q)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3,)

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w,x,y,z)"""
    m00, m01, m02, = R[0,0], R[0,1], R[0,2]
    m10, m11, m12, = R[1,0], R[1,1], R[0,2]
    m20, m21, m22, = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 -m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """unit quaternion (w,x,y,z) -> rotation matrix"""
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z 

    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R 

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quats"""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)  
    dot = np.dot(q0, q1)

    # take shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # nearly linear
        q = q0 + t*(q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    
    theta_0 = np.arccos(dot)
    sin_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_t = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_0
    s1 = sin_t / sin_0
    q = s0*q0 + s1*q1
    return q / (np.linalg.norm(q) + 1e-12)

# Smoothing helpers
def ema_smooth_vec(rows, alpha: float):
    """EMA smoothing for vectors (tvec). rows: list of (idx, vec)"""
    out = {}
    prev = None
    for idx, v in rows:
        if prev is None:
            prev = v.copy()
        else:
            prev = (1 - alpha) * prev + alpha * v
        out[idx] = prev.copy()
    return out

def ema_smooth_quat(rows, alpha: float):
    """EMA-like smoothing for quats via repeated slerp(prev, cur, alpha)."""
    out = {}
    prev = None
    for idx, q in rows:
        if prev is None:
            prev = q.copy()
        else:
            prev = slerp(prev, q, alpha)
        out[idx] = prev.copy()
    return out

def interpolate_gaps(df_full: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """
    Fill small gaps between OK frames:
    - linear interp for tvec
    - slerp for quat
    marks status= 'INTERP'
    """
    df = df_full.copy()
    ok = df["status"].astype(str) == "OK"

    ok_idx = df.index[ok].to_list()
    if len(ok_idx) < 2:
        return df
    
    for a, b in zip(ok_idx[:-1], ok_idx[1:]):
        gap = b - a
        if gap <= 1:
            continue
        if gap - 1 > max_gap:
            continue

        ta = df.loc[a, ["tvec_x","tvec_y","tvec_z"]].to_numpy(dtype=np.float64)
        tb = df.loc[b, ["tvec_x","tvec_y","tvec_z"]].to_numpy(dtype=np.float64)

        qa = df.loc[a, ["qw","qx","qy","qz"]].to_numpy(dtype=np.float64)
        qb = df.loc[b, ["qw","qx","qy","qz"]].to_numpy(dtype=np.float64)

        for k in range(1, gap):
            t = k / gap
            idx = a + k
            if str(df.loc[idx, "status"]) == "OK":
                continue # don't overwrite OK
            tvec = (1 - t) * ta + t * tb
            q = slerp(qa, qb, t)

            df.loc[idx, ["tvec_x","tvec_y","tvec_z"]] = tvec
            df.loc[idx, ["qw","qx","qy","qz"]] = q
            df.loc[idx, "status"] = "INTERP"
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--bad-err", type=float, default=40.0)
    ap.add_argument("--min-tracked", type=int, default=6)

    # Smoothing params
    ap.add_argument("--smooth-window", type=int, default=7, help="EMA window size (~ strength)")
    ap.add_argument("--max-gap", type=int, default=5, help="Max missing frames to interpolate for full smoothing")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Ensure status exists (if older csv)
    if "status" not in df.columns:
        df["status"] = "OK"
        # if rvec missing -> bad
        if df[["rvec_x","rvec_y","rvec_z","tvec_x","tvec_y","tvec_z"]].isna().any(axis=1).any():
            df.loc[df[["rvec_x","rvec_y","rvec_z","tvec_x","tvec_y","tvec_z"]].isna().any(axis=1), "status"] = "BAD"

    # alpha from window (standard EMA relationship)
    w = max(1, int(args.smooth_window))
    alpha = 2.0 / (w + 1.0)

    # Build quat columns for OK frames
    df["qw"] = np.nan
    df["qx"] = np.nan
    df["qy"] = np.nan
    df["qz"] = np.nan

    ok_mask = df["status"].astype(str) == "OK"
    for i in df.index[ok_mask]:
        rvec = df.loc[i, ["rvec_x","rvec_y","rvec_z"]].to_numpy(dtype=np.float64)
        q = rvec_to_quat(rvec)
        df.loc[i, ["qw","qx","qy","qz"]] = q
    

    # A) Smooth only OK frames
    df_ok = df[ok_mask].copy()
    
    t_rows = []
    q_rows = []
    for i in df_ok.index:
        t = df_ok.loc[i, ["tvec_x","tvec_y","tvec_z"]].to_numpy(dtype=np.float64)
        q = df_ok.loc[i, ["qw","qx","qy","qz"]].to_numpy(dtype=np.float64)
        t_rows.append((i, t))
        q_rows.append((i, q))

    t_sm = ema_smooth_vec(t_rows, alpha)
    q_sm = ema_smooth_quat(q_rows, alpha)

    df_ok["tvec_x_sm"] = [t_sm[i][0] for i in df_ok.index]
    df_ok["tvec_y_sm"] = [t_sm[i][1] for i in df_ok.index]
    df_ok["tvec_z_sm"] = [t_sm[i][2] for i in df_ok.index]

    r_sm = [quat_to_rvec(q_sm[i]) for i in df_ok.index]
    df_ok["rvec_x_sm"] = [r[0] for r in r_sm]
    df_ok["rvec_y_sm"] = [r[1] for r in r_sm]
    df_ok["rvec_z_sm"] = [r[2] for r in r_sm]

    out_ok = out_dir / "poses_smoothed_ok.csv"
    df_ok.to_csv(out_ok, index=False)

    # B) Full timeline: interpolate small gaps + smooth
    df_full = df.copy()

    # For OK frames quat already set; for non-OK it is NaN initially
    # Interpolate only between OK frames with <= max_gap
    df_full = interpolate_gaps(df_full, max_gap=int(args.max_gap))

    # now smooth frames where quat & tvec exist (OK or INTERP)
    has_pose = df_full[["tvec_x","tvec_y","tvec_z","qw","qx","qy","qz"]].notna().all(axis=1)
    pose_rows = df_full.index[has_pose].to_list()

    t_rows2, q_rows2 = [], []
    for i in pose_rows:
        t = df_full.loc[i, ["tvec_x","tvec_y","tvec_z"]].to_numpy(dtype=np.float64)
        q = df_full.loc[i, ["qw","qx","qy","qz"]].to_numpy(dtype=np.float64)
        t_rows2.append((i, t))
        q_rows2.append((i, q))

    t_sm2 = ema_smooth_vec(t_rows2, alpha)
    q_sm2 = ema_smooth_quat(q_rows2, alpha)

    df_full["tvec_x_sm"] = np.nan
    df_full["tvec_y_sm"] = np.nan
    df_full["tvec_z_sm"] = np.nan
    df_full["rvec_x_sm"] = np.nan
    df_full["rvec_y_sm"] = np.nan
    df_full["rvec_z_sm"] = np.nan

    for i in pose_rows:
        df_full.loc[i, ["tvec_x_sm","tvec_y_sm","tvec_z_sm"]] = t_sm2[i]
        r = quat_to_rvec(q_sm2[i])
        df_full.loc[i, ["rvec_x_sm","rvec_y_sm","rvec_z_sm"]] = r

    out_full = out_dir / "poses_smoothed.csv"
    df_full.to_csv(out_full, index=False)

    # Quick summary
    n_interp = int((df_full["status"].astype(str) == "INTERP").sum())
    print("✅ Wrote:")
    print(" -", out_ok)
    print(" -", out_full)
    print(f"[INFO] Smoothed OK only: {len(df_ok)} frames")
    print(f"[INFO] Interpolated frames (<=max-gap): {n_interp}")
    print(f"[INFO] EMA alpha={alpha:.4f} (window={w})")

if __name__ == "__main__":
    main()
