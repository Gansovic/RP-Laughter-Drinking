import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

with open("../../laughter_dataset.pkl", "rb") as f:
    X_total, y_total, resumen_data = pickle.load(f)


X_np = np.array(X_total)
y_np = np.array(y_total)

pos_indices = np.where(y_np == 1)[0][:3]
neg_indices = np.where(y_np == 0)[0][:3]

def plot_vector(idx):
    vector = X_np[idx]
    label = "Laughter" if y_np[idx] == 1 else "Non-Laughter"
    plt.figure(figsize=(8, 3))
    plt.plot(vector, marker='o')
    plt.title(f"{label} (Index {idx})")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.show()



print("🔊 Plotting 3 positive (laughter) examples...")
for idx in pos_indices:
    plot_vector(idx)

print("🔈 Plotting 3 negative (non-laughter) examples...")
for idx in neg_indices:
    plot_vector(idx)

counter = Counter(y_total)
plt.bar(["No Laughter", "Laughter"], [counter[0], counter[1]])
plt.title("Label Distribution")
plt.ylabel("Number of Windows")
plt.show()

