import argparse
import re
from pathlib import Path
import pandas as pd

def parse_summary(txt: str) -> dict:
    # Extract numbers from analysis_summary.txt
    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, txt)
        return cast(m.group(1)) if m else default
    
    total = grab(r"Total frames in CSV:\s*(\d+)", int)
    ok = grab(r"Solved frames \(OK\):\s*(\d+)", int)

    mean = grab(r"mean=([0-9.]+)", float)
    median = grab(r"median=([0-9.]+)", float)
    maxv = grab(r"max=([0-9.]+)", float)

    bad = grab(r"Bad frames \(count\):\s*(\d+)", int)
    tracked_mean = grab(r"Tracked points on OK frames:\s*mean=([0-9.]+)", float)

    ok_ratio = (ok / total) if (ok is not None and total) else None

    return dict(
        total_frames=total,
        ok_frames=ok,
        ok_ratio=ok_ratio,
        reproj_mean_px=mean,
        reproj_median_px=median,
        reproj_max_px=maxv,
        bad_frames=bad,
        tracked_mean=tracked_mean,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Project root, e.g. C:\\Users\\shko0\\Bachelorarbeit_CameraTracking")
    ap.add_argument("--out-csv", default="results/comparison/comparison_setA_setB.csv")
    ap.add_argument("--out-tex", default="results/comparison/comparison_setA_setB.tex")
    ap.add_argument("--videos", nargs="+", required=True, help="Video ids like video_02 video_04 ...")
    ap.add_argument("--sets", nargs="+", default=["setA", "setB"], help="Which sets to include")
    ap.add_argument("--summary-rel", default=r"analysis\analysis_summary.txt",
                    help=r"Relative path inside each run folder (default: analysis\analysis_summary.txt)")
    ap.add_argument("--run-rel", default=r"results\video_pose",
                    help=r"Relative base folder from root (default: results\video_pose)")
    args = ap.parse_args()

    root = Path(args.root)
    base = root / args.run_rel

    rows = []
    for vid in args.videos:
        for s in args.sets:
            # For video_02 not have setB
            summary_path = base / vid / s / Path(args.summary_rel)
            if not summary_path.exists():
                # allow missing (e.g., video_02 setB)
                continue
            txt = summary_path.read_text(encoding="utf-8", errors="ignore")
            d = parse_summary(txt)
            d.update(video_id=vid, set=s)
            rows.append(d)

    df = pd.DataFrame(rows)
    out_csv = root / args.out_csv
    out_tex = root / args.out_tex
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Nice ordering
    df["video_num"] = df["video_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["video_num", "set"]).drop(columns=["video_num"])

    df.to_csv(out_csv, index=False)

    # LaTeX table
    df_tex = df.copy()
    # format numbers
    for col in ["ok_ratio","reproj_mean_px","reproj_median_px","reproj_max_px","tracked_mean"]:
        if col in df_tex.columns:
            df_tex[col] = df_tex[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    latex = df_tex.to_latex(index=False, escape=True)
    out_tex.write_text(latex, encoding="utf-8")

    print("✅ Wrote:")
    print("CSV:", out_csv)
    print("TEX:", out_tex)

if __name__ == "__main__":
    main()