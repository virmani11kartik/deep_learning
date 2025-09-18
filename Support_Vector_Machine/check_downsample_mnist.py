from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2

def downsample_cv2(X, new_size=(14,14)):
    # X shape: (N, 784). Return shape: (N, 196)
    X2 = []
    for img in X:
        a = img.reshape(28, 28)
        b = cv2.resize(a, new_size, interpolation=cv2.INTER_AREA)  # good for shrinking
        X2.append(b.ravel())
    return np.array(X2, dtype=np.float32)

# Load dataset (from cache if already downloaded)
ds = fetch_openml('mnist_784', as_frame=False)
X_all = ds.data.astype(np.float32)      # (70000, 784)
y_all = ds.target.astype(int) 

print("Before Downsampling")
print("Data shape:", X_all.shape)
print("Labels shape:", y_all.shape)

# Train/test split
x, x_test, y, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

x14      = downsample_cv2(x)            # (56000, 196)
x_test14 = downsample_cv2(x_test)       # (14000, 196)

print("After Downsampling")
print("Data shape:", x14.shape, x_test14.shape)

# Show one image
a = x14[0].reshape((14, 14))
plt.imshow(a, cmap="gray")
plt.title(f"Label: {y[0]}")
plt.show()