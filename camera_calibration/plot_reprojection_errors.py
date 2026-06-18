import csv
import matplotlib.pyplot as plt

# Pfad zur CSV-Datei
csv_path = r"C:\Users\shko0\Bachelorarbeit_CameraTracking\camera_calibration\calib_out03\logs\per_image_errors.csv"

indices = []
errors = []

# CSV einlesen
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        indices.append(int(row["index"]))
        errors.append(float(row["reproj_error(px)"]))

# Plot erstellen
plt.figure(figsize=(10, 4))
plt.plot(indices, errors, marker="o", linestyle="-")
plt.xlabel("Bildindex")
plt.ylabel("Reprojektionsfehler [Pixel]")
plt.title("Bildweiser Reprojektionsfehler der Kamerakalibrierung")
plt.grid(True)

# RMS-Linie
rms = sum(e*e for e in errors) / len(errors)
plt.axhline(y=(rms**0.5), color="red", linestyle="--", label="RMS")
plt.legend()

#Speichern
plt.tight_layout()
plt.savefig("reprojection_errors.png", dpi=300)
plt.show()