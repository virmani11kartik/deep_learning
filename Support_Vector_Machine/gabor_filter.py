# hw1_part_h_gabor_svm.py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np
from skimage.filters import gabor_kernel, gabor
import time
import os

# -------------------------
# Config
# -------------------------
RNG = np.random.RandomState(42)
SAVE_DIR = "./outputs_part_h"
os.makedirs(SAVE_DIR, exist_ok=True)

USE_PCA = True          # Set False to skip PCA
PCA_COMPONENTS = 512    # Reasonable compression (tune if needed)

# -------------------------
# Utilities
# -------------------------
def downsample_blockavg(X28x28):
    """Downsample 28x28 images to 14x14 using 2x2 block averaging (fast, no cv2)."""
    N = X28x28.shape[0]
    A = X28x28.reshape(N, 28, 28)
    B = A.reshape(N, 14, 2, 14, 2).mean(axis=(2, 4))
    return B.reshape(N, 196).astype(np.float32)

def sample_per_class(X, y, per_class=200):
    """
    Sample 'per_class' examples per class uniformly at random.
    We'll later split 50/50 into train/val so that we end up with 100/class each.
    """
    xs, ys = [], []
    for c in range(10):
        idx = np.where(y == c)[0]
        pick = RNG.choice(idx, size=per_class, replace=False)
        xs.append(X[pick])
        ys.append(y[pick])
    X_sub = np.vstack(xs)
    y_sub = np.concatenate(ys)
    perm = RNG.permutation(len(y_sub))
    return X_sub[perm], y_sub[perm]

def create_gabor_filters():
    """Create a bank of 36 Gabor filters (4 thetas × 3 freqs × 3 bandwidths)."""
    filters = []
    thetas = np.arange(0, np.pi, np.pi/4)        # 0, 45, 90, 135  (4)
    freqs  = np.arange(0.05, 0.5, 0.15)          # 0.05, 0.20, 0.35 (3)
    bands  = np.arange(0.3, 1.0, 0.3)            # 0.3, 0.6, 0.9    (3)
    for f in freqs:
        for th in thetas:
            for bw in bands:
                kernel = gabor_kernel(frequency=f, theta=th, bandwidth=bw)
                filters.append((kernel, f, th, bw))
    print(f"Created {len(filters)} Gabor filters")  # should be 36
    return filters

def extract_gabor_features(X14x14, gabor_filters):
    """
    For each image (14x14), convolve with each Gabor filter, take REAL response,
    flatten 14x14 and concatenate across all filters -> 7056-D per image.
    """
    n_samples = X14x14.shape[0]
    H = W = 14
    n_filters = len(gabor_filters)
    feats = np.zeros((n_samples, H * W * n_filters), dtype=np.float32)

    print("Extracting Gabor features...")
    t0 = time.time()
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  image {i}/{n_samples}")
        img = X14x14[i].reshape(H, W)
        offset = 0
        for (kernel, freq, theta, bw) in gabor_filters:
            real_resp, _ = gabor(img, frequency=freq, theta=theta, bandwidth=bw)
            rflat = real_resp.astype(np.float32).ravel()
            feats[i, offset:offset + H * W] = rflat
            offset += H * W
    print(f"Feature extraction done in {time.time()-t0:.2f}s")
    return feats

def visualize_gabor_filters_and_responses(gabor_filters, sample_image14x14, save_path):
    """Visualize 4 example filters (real/imag) and their responses on a sample 14x14 image."""
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle("Gabor Filters and Responses (examples)")

    sample_img = sample_image14x14.reshape(14, 14)

    for i in range(4):
        kernel, freq, theta, bw = gabor_filters[i]
        axes[0, i].imshow(kernel.real, cmap='gray')
        axes[0, i].set_title(f'Filter {i+1}\nf={freq:.2f}, θ={theta*180/np.pi:.0f}°, bw={bw:.1f}')
        axes[0, i].axis('off')

        axes[1, i].imshow(kernel.imag, cmap='gray')
        axes[1, i].set_title('Imag part')
        axes[1, i].axis('off')

        real_response, _ = gabor(sample_img, frequency=freq, theta=theta, bandwidth=bw)
        axes[2, i].imshow(real_response, cmap='gray')
        axes[2, i].set_title(f'Response (mean={np.mean(real_response):.3f})')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")

# -------------------------
# Main
# -------------------------
def main():
    print("Loading MNIST dataset...")
    ds = fetch_openml('mnist_784', as_frame=False)
    X_all = ds.data.astype(np.float32) / 255.0    # normalize to [0,1]
    y_all = ds.target.astype(np.int64)

    # Sample 200/class, then split to 100/class train + 100/class val
    print("Sampling dataset (200/class) and splitting 50/50...")
    X_sample, y_sample = sample_per_class(X_all, y_all, per_class=200)

    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.5, random_state=42, stratify=y_sample
    )
    print("Shapes before downsample:", X_train.shape, X_val.shape)

    # Downsample to 14x14
    X_train_ds = downsample_blockavg(X_train)  # (N, 196)
    X_val_ds   = downsample_blockavg(X_val)    # (N, 196)
    print("Shapes after downsample:", X_train_ds.shape, X_val_ds.shape)

    # Build Gabor filter bank (36 filters)
    gabor_filters = create_gabor_filters()

    # Visualize a few filters + responses
    visualize_gabor_filters_and_responses(
        gabor_filters, X_train_ds[0], os.path.join(SAVE_DIR, "gabor_filters_visualization.png")
    )

    # Extract FULL Gabor features (7056-D)
    X_train_gabor = extract_gabor_features(X_train_ds, gabor_filters)
    X_val_gabor   = extract_gabor_features(X_val_ds,   gabor_filters)
    print("Gabor feature shapes:", X_train_gabor.shape, X_val_gabor.shape)

    # -------------------------
    # Pipelines
    # -------------------------
    # Gabor pipeline: Standardize -> (optional PCA) -> SVC(RBF)
    steps = [('scaler', StandardScaler(with_mean=True))]
    if USE_PCA:
        steps.append(('pca', PCA(n_components=PCA_COMPONENTS, random_state=42)))
    steps.append(('svc', svm.SVC(C=0.6, kernel='rbf', gamma='scale')))
    clf_gabor = Pipeline(steps)

    # Baseline pipeline on raw downsampled pixels: Standardize -> SVC(RBF)
    baseline = Pipeline([
        ('scaler', StandardScaler(with_mean=True)),
        ('svc', svm.SVC(C=0.8, kernel='rbf', gamma='scale')),
    ])

    # -------------------------
    # Train & Evaluate (Gabor)
    # -------------------------
    print("\nTraining SVM with Gabor features...")
    t0 = time.time()
    clf_gabor.fit(X_train_gabor, y_train)
    train_time = time.time() - t0

    y_pred_gabor = clf_gabor.predict(X_val_gabor)
    val_acc_gabor = accuracy_score(y_val, y_pred_gabor)
    val_err_gabor = 1.0 - val_acc_gabor
    print(f"Gabor SVM - Validation accuracy: {val_acc_gabor:.4f} (error: {val_err_gabor:.4f})")
    print(f"Gabor SVM - Train time: {train_time:.2f}s")

    # Try to report support vector ratio when the last step is SVC
    svc_gabor = clf_gabor.named_steps['svc']
    sv_ratio_gabor = len(svc_gabor.support_) / len(X_train_gabor)
    print(f"Gabor SVM - Support vectors: {len(svc_gabor.support_)} / {len(X_train_gabor)} "
          f"(ratio: {sv_ratio_gabor:.3f})")

    # -------------------------
    # Train & Evaluate (Baseline)
    # -------------------------
    print("\nTraining baseline SVM with raw 14x14 pixels...")
    t0 = time.time()
    baseline.fit(X_train_ds, y_train)
    train_time = time.time() - t0

    y_pred_baseline = baseline.predict(X_val_ds)
    val_acc_baseline = accuracy_score(y_val, y_pred_baseline)
    val_err_baseline = 1.0 - val_acc_baseline
    print(f"Baseline SVM - Validation accuracy: {val_acc_baseline:.4f} (error: {val_err_baseline:.4f})")
    print(f"Baseline SVM - Train time: {train_time:.2f}s")

    svc_base = baseline.named_steps['svc']
    sv_ratio_baseline = len(svc_base.support_) / len(X_train_ds)
    print(f"Baseline SVM - Support vectors: {len(svc_base.support_)} / {len(X_train_ds)} "
          f"(ratio: {sv_ratio_baseline:.3f})")

    # -------------------------
    # Confusion matrices & comparison
    # -------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cm_gabor = confusion_matrix(y_val, y_pred_gabor, labels=np.arange(10))
    disp_gabor = ConfusionMatrixDisplay(confusion_matrix=cm_gabor, display_labels=np.arange(10))
    disp_gabor.plot(ax=ax1, values_format='d', colorbar=False)
    ax1.set_title(f"Gabor SVM (Acc: {val_acc_gabor:.3f})")

    cm_baseline = confusion_matrix(y_val, y_pred_baseline, labels=np.arange(10))
    disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=np.arange(10))
    disp_baseline.plot(ax=ax2, values_format='d', colorbar=False)
    ax2.set_title(f"Baseline SVM (Acc: {val_acc_baseline:.3f})")

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, "gabor_vs_baseline_confusion.png")
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved {fig_path}")

    print(f"\nAccuracy improvement (Gabor - Baseline): {val_acc_gabor - val_acc_baseline:+.4f} points")

if __name__ == "__main__":
    main()
