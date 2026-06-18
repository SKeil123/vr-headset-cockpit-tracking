import matplotlib.pyplot as plt
import numpy as np


# Median Δθ Werte (Videos 04,05,06,08,09,10,12,15,19)
median_A = [3.787, 2.434, 5.319, 1.884, 7.590, 1.613, 1.518, 2.051, 1.903]
median_B = [1.345, 1.890, 1.402, 2.019, 7.590, 1.963, 1.364, 1.714, 1.903]

plt.figure()
plt.boxplot([median_A, median_B])
plt.xticks([1, 2], ["Set A", "Set B"])
plt.ylabel("Median Δθ (deg)")
plt.title("Boxplot of Median Δθ per Video")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/comparison/boxplot_median_dtheta.png", dpi=300)
plt.show()


p95_A = [89.268, 16.347, 102.042, 10.556, 31.398, 13.224, 27.314, 3.578, 3.406]
p95_B = [19.173, 3.208, 3.447, 9.414, 31.398, 3.268, 2.624, 3.473, 3.406]

avg_A = np.mean(p95_A)
avg_B = np.mean(p95_B)

plt.figure()
plt.bar(["Set A", "Set B"], [avg_A, avg_B])
plt.ylabel("Average p95 Δθ (deg)")
plt.title("Average p95 Δθ Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/comparison/barplot_p95_dtheta.png", dpi=300)
plt.show()