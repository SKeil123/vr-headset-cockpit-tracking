import cv2
import json
import numpy as np
from pathlib import Path
import argparse


# ------ Konfiguration ------
POINTS = [
    ("P1", "Zentrum Airspeed Indicator"),
    ("P2", "Zentrum Turn Coordinator"),
    ("P3", "Zentrum Vertical Speed Indicator (VSI)"),
    ("P4", "Zentrum Kompass / Heading Indicator"),
    ("P5", "Schraube oben links am Airspeed"),
    ("P6", "Schraube oben rechts am Airspeed"),
    ("P7", "Schraube unten links am VSI"),
    ("P8", "Schraube unten rechts am VSI"),
    ("P9", "Schraube oben rechts am Kompass"),
    ("P10", "Schraube unten links am Turn Coordinator"),
]

WINDOW = "Pick Points (Wheel=Zoom, RightDrag=Pan, LeftClick=Set Point, Backspace=Undo, S=Save)"
OUTPUT_PIXELS_JSON = "cockpit_points_pixels_video_02.json"
OUTPUT_3D_JSON = "cockpit_points_3d.json"
SCALE_MM_PER_PX = None

# Skalierung:
# Option 1 : man klickt zwei Punkte (z.B. Durchmesser eines Instruments) und gibt die echte Distanz in mm ein.
# Option 2 : setze SCALE_MM_PER_PX direkt (falls man sie schon kennt). Dann wird kein Scale-Klick benötigt.


# ------ Interaktiver Zustand ------
state = {
    "zoom": 1.0,
    "pan": np.array([0.0, 0.0], dtype=np.float32),   # in Bildschirm-Pixeln
    "dragging": False,
    "drag_start": (0, 0),
    "pan_start": np.array([0.0, 0.0], dtype=np.float32),
    "picked": {},              # {"P1": (u,v), ...} in Originalbild-Pixeln
    "idx": 0,                  # nächster Punktindex
    "scale_clicks": [],        # [(u,v), (u,v)] für Skalierung
}

def clamp_zoom(z):
    return float(np.clip(z, 0.2, 8.0))

def screen_to_image(x, y, img_w, img_h):
    """
    Mappt Bildschirmkoordinaten (Window) -> Originalbildkoordinaten (u,v)
    mit aktuellem Zoom & Pan.
    """
    z = state["zoom"]
    pan = state["pan"]
    u = (x - pan[0]) / z
    v = (y - pan[1]) / z
    # Begrenzen auf Bildbereich
    u = float(np.clip(u, 0, img_w - 1))
    v = float(np.clip(v, 0, img_h - 1))
    return (u, v)

def draw_overlay(base_img):
    """
    Zeichnet Labels + Punkte auf eine gerenderte Ansicht (nach Zoom/Pan).
    """
    img_h, img_w = base_img.shape[:2]
    z = state["zoom"]
    pan = state["pan"]

    # Render view (simple affine scale + translate)
    M = np.array([[z, 0, pan[0]],
                 [0, z, pan[1]]], dtype=np.float32)
    view = cv2.warpAffine(base_img, M, (int(img_w * z + abs(pan[0]) + 2000), int(img_h * z + abs(pan[1]) + 2000)))

    # Wir schneiden das View-Bild auf eine fixe Fenstergröße (dynamisch anhand screen size)
    # OpenCV hat keine direkte Window-size API - wir nehmen eine sinnvolle Maximalgröße.
    # Man kann das Fenster später manuell größer ziehen.
    win_w, win_h = 1400, 900
    view = view[:win_h, :win_w].copy()

    # Cureent instruction text
    if state["idx"] < len(POINTS):
        pid, desc = POINTS[state["idx"]]
        txt = f"Next: {pid} - {desc}"
    else:
        txt = "All points selecte. Press S to save. (Optional: pick scale if asked.)"
    cv2.rectangle(view, (0, 0), (win_w, 70), (0, 0, 0), -1)
    cv2.putText(view, txt, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw already picked points
    for pid, (u, v) in state["picked"].items():
        sx = int(u * z + pan[0])
        sy = int(v * z + pan[1])
        if 0 <= sx < win_w and 0 <= sy < win_h:
            cv2.circle(view, (sx, sy), 6, (0, 255, 0), -1)
            cv2.putText(view, pid, (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw scale clicks if any
    for i, (u, v) in enumerate(state["scale_clicks"]):
        sx = int(u * z + pan[0])
        sy = int(v * z + pan[1])
        if 0 <= sx < win_w and 0 <= sy < win_h:
            cv2.circle(view, (sx, sy), 6, (255, 0, 255), -1)
            cv2.putText(view, f"S{i+1}", (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    return view

def mouse_cb(event, x, y, flags, param):
    base_img = param["img"]
    img_h, img_w = base_img.shape[:2]

    # Zoom with mouse wheel (OpenCV: flags carries wheel delta)
    if event == cv2.EVENT_MOUSEHWHEEL:
        delta = 1 if flags > 0 else -1
        old_zoom = state["zoom"]
        new_zoom = clamp_zoom(old_zoom * (1.1 if delta > 0 else 1/1.1))

        # Zoom around cursor: keep image point under cursor stable
        u, v = screen_to_image(x, y, img_w, img_h)
        # New pan to keep (x,y) mapped to same (u,v)
        state["zoom"] = new_zoom
        state["pan"][0] = x - u * new_zoom
        state["pan"][1] = y - v * new_zoom

    # Right-button drag = pan
    elif event == cv2.EVENT_RBUTTONDOWN:
        state["dragging"] = True
        state["drag_start"] = (x, y)
        state["pan_start"] = state["pan"].copy()
    elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
        dx = x - state["drag_start"][0]
        dy = y - state["drag_start"][1]
        state["pan"] = state["pan_start"] + np.array([dx, dy], dtype=np.float32)
    elif event == cv2.EVENT_RBUTTONUP:
        state["dragging"] = False

    # Left click = set next point (or scale point if needed)
    elif event == cv2.EVENT_LBUTTONDOWN:
        u, v = screen_to_image(x, y, img_w, img_h)

        # If we still need scale clicks
        if param["need_scale_clicks"] and len(state["scale_clicks"]) < 2:
            state["scale_clicks"].append((u, v))
            print(f"[SCALE] Click {len(state['scale_clicks'])}/2 at (u,v)=({u:.2f},{v:.2f})")
            return
        
        # Otherwise pick next point
        if state["idx"] < len(POINTS):
            pid, desc = POINTS[state["idx"]]
            state["picked"][pid] = (u, v)
            print(f"[PICK] {pid} ({desc}) -> (u,v)=({u:.2f},{v:.2f})")
            state["idx"] += 1

def compute_scale_mm_per_px():
    """
    Returns scale in mm/px either from constant or from two clicked scale points.
    """
    global SCALE_MM_PER_PX 
    if SCALE_MM_PER_PX is not None:
        return float(SCALE_MM_PER_PX)
    
    if len(state["scale_clicks"]) < 2:
        return None
    
    (u1, v1), (u2, v2) = state["scale_clicks"]
    dist_px = float(np.hypot(u2 - u1, v2 - v1))
    if dist_px < 1e-6:
        return None
    
    # Ask user for real-world distance in mm
    while True:
        try:
            mm = float(input("\nEnter real distance between S1 and S2 in mm (e.g. instrument diameter): ").strip())
            if mm <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number like 80 or 76.5")

    s = mm / dist_px
    print(f"[SCALE] dist_px={dist_px:.4f} px, dist_mm={mm:.4f} mm => scale={s:.6f} mm/px\n")
    return s

def save_results(out_dir: Path, scale_mm_per_px: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save pixels
    pixels = {
        "image": str(out_dir / "reference_frame.png"),
        "points": [
            {"id": pid, "name": desc, "uv": [float(state["picked"][pid][0]), float(state["picked"][pid][1])]}
            for pid, desc in POINTS
        ]
    }
    (out_dir / OUTPUT_PIXELS_JSON).write_text(json.dumps(pixels, indent=2), encoding="utf-8")

    # Compute 3D (planar, origin at P1)
    u0, v0 = state["picked"]["P1"]
    pts3d = []
    for pid, desc in POINTS:
        u, v = state["picked"][pid]
        x = (u - u0) * scale_mm_per_px
        y = (v0 - v) * scale_mm_per_px     # flip y-axis (image down -> world up)
        z = 0.0
        pts3d.append({"id": pid, "name": desc, "xyz": [float(x), float(y), float(z)]})
    
    out3d = {
        "unit": "mm (scaled)" if SCALE_MM_PER_PX is None else "mm",
        "coordinate_system": {
            "origin": "P1",
            "origin_definition": "P1 = Zentrum Airspeed Indicator",
            "x_axis": "right (panel)",
            "y_axis": "up (panel)",
            "z_axis": "out_of_panel (towards camera)",
            "planar_assumption": "all points z=0"
        },
        "scale": {
            "mm_per_px": float(scale_mm_per_px),
            "method": "two-click scale (S1,S2)" if SCALE_MM_PER_PX is None else "fixed constant"
        },
        "points": pts3d
    }
    (out_dir / OUTPUT_3D_JSON).write_text(json.dumps(out3d, indent=2), encoding="utf-8")

    print(f"✅ Saved:\n - {out_dir / OUTPUT_PIXELS_JSON}\n - {out_dir / OUTPUT_3D_JSON}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to reference frame image (png/jpg)")
    ap.add_argument("--out", default="cockpit_geometry", help="Output directory")
    ap.add_argument("--need-scale", action="store_true",
                    help="If set: you must click two scale points S1, S2 and enter mm distance to compute mm/px.")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError("Could not read image.")
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save reference frame copy for documentation
    cv2.imwrite(str(out_dir / "reference_frame.png"), img)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1400, 900)

    param = {"img": img, "need_scale_clicks": bool(args.need_scale)}

    cv2.setMouseCallback(WINDOW, mouse_cb, param)

    print("\n--- Instructions ---")
    print("Mouse wheel: zoom")
    print("Right mouse drag: pan")
    print("Left click: set next point (P1..P10)")
    print("Backspace: undo last point")
    print("S: save (only after all points selected)")
    if args.need_scale:
        print("Scale: first click two points S1, S2 (magenta) THEN you'll enter their real mm distance in console.\n ")
    else:
        print("Scale: using fixed SCALE_MM_PER_PX in script (set it a top) OR run with --need-scale.\n")

    while True:
        view = draw_overlay(img)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF

        # Undo
        if key == 8:    # Backspace
            if state["idx"] > 0:
                state["idx"] -= 1
                pid, _ = POINTS[state["idx"]]
                if pid in state["picked"]:
                    del state["picked"][pid]
                print(f"[UNDO] Removed {pid}")
            elif len(state["scale_clicks"]) > 0:
                state["scale_clicks"].pop()
                print("[UNDO] Removed last scale click")

        # Save
        elif key in (ord('s'), ord('S')):
            if state["idx"] < len(POINTS):
                print("⚠️ Not all points selected yet.")
                continue

            scale = compute_scale_mm_per_px()
            if scale is None:
                print("⚠️ Scale not available. Either set SCALE_MM_PER_PX at top or run with --need-scale and click S1,S2.")
                continue

            save_results(out_dir, scale)
            break

        # Quit
        elif key in (27, ord('q'), ord('Q')):   # Esc or Q
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()