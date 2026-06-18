import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Analyze poses.csv(reproj over time, tracked points, bad frames, summary).")
    ap.add_argument("--csv", type=str, required=True, help="Path to poses.csv")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory (default: <csv_dir>/analysis)")
    ap.add_argument("--bad-err", type=float, default=40.0, help="Reprojection error threshold for 'bad' frames")
    ap.add_argument("--min-tracked", type=int, default=6, help="Tracked points threshold for 'bad' frames")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if args.out_dir is None:
        out_dir = csv_path.parent / "analysis"
    else:
        out_dir = Path(args.out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Robust: ensure colums exist (older csv fallback)
    if "status" not in df.columns:
        df["status"] = "OK"
    if "n_inliers" not in df.columns:
        df["n_inliers"] = np.nan
    
    # Coerce numeric colums (NaN-safe)
    num_cols = ["n_tracked", "n_inliers", "reproj_err_px",
                "rvec_x", "rvec_y", "rvec_z", "tvec_x", "tvec_y", "tvec_z"]
    
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Define "OK" frames (pose exists)
    ok_mask = (df["status"] == "OK") & df["reproj_err_px"].notna()
    df_ok = df[ok_mask].copy()

    # Define "bad" frames:
    # 1) explicitly marked non-OK
    # 2) OR reproj too high
    # 3) OR too few tracked points
    bad_mask = (~ok_mask) | (df["reproj_err_px"] > args.bad_err) | (df["n_tracked"] < args.min_tracked)
    df_bad = df[bad_mask].copy()

    # Summary statistics (only on OK frames)
    solved = int(len(df_ok))
    total = int(len(df))

    if solved > 0:
        err_mean = float(df_ok["reproj_err_px"].mean())
        err_med = float(df_ok["reproj_err_px"].median())
        err_max = float(df_ok["reproj_err_px"].max())

        tr_mean = float(df_ok["n_tracked"].mean())
        tr_min = int(df_ok["n_tracked"].min())
        tr_max = int(df_ok["n_tracked"].max())
    else:
        err_mean = err_med = err_max = np.nan
        tr_mean = np.nan
        tr_min = tr_max = 0

    # Write bad frames list
    bad_csv = out_dir / "bad_frames.csv"
    # Keep important columns in a stable order (only those that exist)
    cols_out = ["frame_idx", "frame_name", "n_tracked", "n_inliers", "reproj_err_px", "status"]
    cols_out += [c for c in ["rvec_x", "rvec_y", "rvec_z", "tvec_x", "tvec_y", "tvec_z"] if c in df_bad.columns]
    df_bad[cols_out].to_csv(bad_csv, index=False)

    # Write analysis summary
    summary_txt = out_dir / "analysis_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"CSV: {csv_path}\n")
        f.write(f"Total frames in CSV: {total}\n")
        f.write(f"Solved frames (OK): {solved}\n\n")

        f.write(f"Reproj error (px) on OK frames: mean={err_mean:.2f}, median={err_med:.2f}, max={err_max:.2f}\n")
        if solved > 0:
            f.write(f"Tracked points on OK frames: mean={tr_mean:.2f}, min={tr_min}, max={tr_max}\n")
        else:
            f.write("Tracked points on OK frames: n/a\n")

        f.write("\nBad frame rule: \n")
        f.write(f"- status != OK OR reproj_err_px > {args.bad_err} OR n_tracked < {args.min_tracked}\n")
        f.write(f"Bad frames (count): {int(len(df_bad))}\n")
        if len(df_bad) > 0:
            f.write("Bad frame indices: " + ", ".join(map(str, df_bad["frame_idx"].astype(int).tolist())) + "\n")
    
    # Plot: reprojection error over time (OK curve) + mark bad frames
    # Use full frame_idx axis (so gaps are visible)
    x_all = df["frame_idx"].to_numpy()
    y_err = df["reproj_err_px"].to_numpy()

    # Plot err (including NaNs - Matplotlib will break the line, which is good)
    plt.figure()
    plt.plot(x_all, y_err)
    plt.title("Reprojection error over time")
    plt.xlabel("frame_idx")
    plt.ylabel("reproj_err_px")

    # Mark bad frames as vertical lines (thin)
    for bx in df_bad["frame_idx"].dropna().astype(int).tolist():
        plt.axvline(bx)

    plt.tight_layout()
    reproj_png = out_dir / "reproj_over_time.png"
    plt.savefig(reproj_png, dpi=150)
    plt.close()

    # Plot: tracked points over time + mark bad frames
    plt.figure()
    plt.plot(x_all, df["n_tracked"].to_numpy())
    plt.title("Tracked points over time")
    plt.xlabel("frame_idx")
    plt.ylabel("n_tracked")

    for bx in df_bad["frame_idx"].dropna().astype(int).tolist():
        plt.axvline(bx)

    plt.tight_layout()
    tracked_png = out_dir / "tracked_over_time.png"
    plt.savefig(tracked_png, dpi=150)
    plt.close()

    print("✅ Analysis done")
    print("Summary:", summary_txt)
    print("Bad frames:", bad_csv)
    print("Plots:", reproj_png, tracked_png)

if __name__ == "__main__":
    main()   
