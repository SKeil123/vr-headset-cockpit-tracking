# Recomend_winSize_from_image
# Vorgehen :
# 1) Bild laden -> Graustufen
# 2) Checkerboard Koordinaten robust finden (SB -> Fallback classic)
# 3) Pixelabstände benachbarter Ecken (horizontal/vertikal) messen
# 4) Typische Kantenlänge in Pixeln schätzen (Median) = "px_per_square"
# 5) winSize Empfehlung anhand Schwellen ableiten
# 6) (optional) Visualisierung mit Ecken + Messungen speichern

import argparse 
from pathlib import Path
import sys
import numpy as np
import cv2


# Hilfsfunktionen
def imread_any(path: Path):
    """Lesen ein Bild robust (unterstützt auch Pfade mit Sonderzeichen)."""
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def find_corners_robust(gray, pattern_size):
    """Versuchen zuerst SB (robust), dann klassische Methode mit Subpixel Refinement."""
    # 1) SB Detektor
    try:
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if ok:
            return True, corners.astype(np.float32)
    except Exception:
        pass

    # 2) Klassischer Detektor + Subpixel Refinement
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return True, corners.astype(np.float32)

def estimate_square_pixels(corners, pattern_size):
    """Schätzen typische Pixel Länge pro Checker Square via Nachbarabstände.
    corners: (N, 1, 2) float32 (OpenCV Format)
    pattern_Size: (cols, rows) = Anzahl INNERER Ecken
    Rückgabe: median_px (float), horiz_list, vert_list
    """
    cols, rows = pattern_size
    pts = corners.reshape(-1, 2)  # (N,2)

    # Indizes: Zeilenweise Ordnung 
    # Horizontal: Abstand zwischen (c,r) und (c+1, r)
    horiz = []
    for r in range(rows):
        for c in range(cols - 1):
            i = r * cols + c
            j = r * cols + (c + 1)
            d = np.linalg.norm(pts[j] - pts[i])
            horiz.append(d)

    # Vertikal: Abstand zwischen (c,r) und (c, r+1)
    vert = []
    for r in range(rows - 1):
        for c in range(cols):
            i = r * cols + c
            j = (r + 1) * cols + c
            d = np.linalg.norm(pts[j] - pts[i])
            vert.append(d)

    horiz = np.array(horiz, dtype=np.float32)
    vert = np.array(vert, dtype=np.float32)

    # Robuste Schätzung via Median (Ausreißer resistent)
    all_edges = np.concatenate([horiz, vert])
    median_px = float(np.median(all_edges))
    return median_px, horiz, vert

def recommend_win_size(px_per_square):
    """Leiten eine cornerSubPix winSize-Empfehlung aus px/Square ab.
    Mapping orientiert sich an praxisnahen Ranges.
    Rückgabe: (w, h), info_text
    """
    p = px_per_square
    if p <= 20:
        ws = (5, 5)
        note = "very small squares in image"
    elif p <= 35:
        ws = (7, 7)
        note = "small squares"
    elif p <= 80:
        ws = (9, 9)
        note = "medium squares"
    elif p <= 150:
        ws = (11, 11)
        note = "medium-large squares"
    elif p <= 250:
        ws = (13, 13)
        note = "large squares"
    else:
        ws = (15, 15)
        note = "very large squares"
    info = f"Heuristic bucket: {note}"
    return ws, info

def draw_visualization(img, pattern_size, corners, horiz, vert, out_path: Path):
    """Zeichnen Ecken + ein paar Besipielkantenlängen und speichert PNG."""
    vis = img.copy()
    cols, rows = pattern_size

    # Ecken zeichnen
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    # Beispielhaft einige horizontale & vertikale Abstände einzeichnen
    pts = corners.reshape(-1, 2).astype(int)
    color_h = (0, 255, 255)
    color_v = (255, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Horizontal: erste Reihe
    r = 0
    for c in range(min(cols - 1, 4)):
        i = r * cols + c
        j = r * cols + (c + 1)
        p1 = tuple(pts[i])
        p2 = tuple(pts[j])
        cv2.line(vis, p1, p2, color_h, 2)
        d = np.linalg.norm(pts[j] - pts[i])
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(vis, f"{d:.1f}px", mid, font, 0.6, color_v, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


# Hauptfunktion
def main():
    ap = argparse.ArgumentParser(description="Recommend cornerSubPix winSize based on a sample" \
    "chekcerboard image.")
    ap.add_argument("--image", type=str, required=True, help="Path to a checkerboard image")
    ap.add_argument("--pattern-cols", type=int, required=True, help="Number of inner corners horizontally")
    ap.add_argument("--pattern-rows", type=int, required=True, help="Number of inner corners vertically")
    ap.add_argument("--save-vis", type=str, default="", help="Optional path to save a visualization PNG")
    args = ap.parse_args()

    img_path = Path(args.image)
    img = imread_any(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}", file=sys.stderr)
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    ok, corners = find_corners_robust(gray, pattern_size)
    if not ok:
        print("[ERROR] Could not detect checkerboard corners. Check pattern size or image quality.", file=sys.stderr)
        sys.exit(2)
    px_per_square, horiz, vert = estimate_square_pixels(corners, pattern_size)
    ws, info = recommend_win_size(px_per_square)

    print("[INFO] Estimated pixel length per square (median): {:.2f} px".format(px_per_square))
    print("[SUGGESTION] Recommended winSize for cornerSubPix: {} (i.e., +/-{} px search)".format(ws, ws[0]))
    print(f"[NOTE] {info}")

    if args.save_vis:
        out_path = Path(args.save_vis)
        try:
            draw_visualization(img, pattern_size, corners, horiz, vert, out_path)
            print(f"[SAVE] Visualization saved to: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save visualization: {e}")

if __name__ == "__main__":
    main()