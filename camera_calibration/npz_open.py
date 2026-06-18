import numpy as np

dateipfad = r"C:\Users\shko0\Bachelorarbeit_CameraTracking\camera_calibration\calib_out03\intrinsics.npz"
data = np.load(dateipfad, allow_pickle=True)
print("Enthaltene Keys:")
print(data.files)

print("\n--- Inhalte ---")
for k in data.files:
    print(f"\n{k}:")
    print(data[k])
