from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from skimage.filters import gabor_kernel, gabor
import time

RNG = np.random.RandomState(42)

def downsample_blockavg(X28x28):
    """Downsample 28x28 images to 14x14 using block averaging"""
    N = X28x28.shape[0]
    A = X28x28.reshape(N, 28, 28)
    B = A.reshape(N, 14, 2, 14, 2).mean(axis=(2, 4))
    return B.reshape(N, 196).astype(np.float32)

def sample_per_class(X, y, per_class=100):
    """Sample a specified number of examples per class"""
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
    """Create a bank of Gabor filters with different orientations and frequencies"""
    filters = []
    
    # Different orientations (0, 45, 90, 135 degrees)
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Different frequencies
    frequencies = [0.1, 0.2, 0.3]
    
    # Fixed bandwidth
    bandwidth = 1
    
    for freq in frequencies:
        for theta in thetas:
            kernel = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)
            filters.append((kernel, freq, theta))
    
    print(f"Created {len(filters)} Gabor filters")
    return filters

def extract_gabor_features(X, gabor_filters):
    """Extract Gabor filter responses for each image"""
    n_samples = X.shape[0]
    n_filters = len(gabor_filters)
    
    # Each filter will produce one feature (mean of real part)
    features = np.zeros((n_samples, n_filters))
    
    print("Extracting Gabor features...")
    for i in range(n_samples):
        if i % 200 == 0:
            print(f"Processing image {i}/{n_samples}")
            
        # Reshape to 14x14 image
        image = X[i].reshape(14, 14)
        
        for j, (kernel, freq, theta) in enumerate(gabor_filters):
            # Apply Gabor filter - use only real part
            real_response, _ = gabor(image, frequency=freq, theta=theta, bandwidth=1)
            
            # Use mean response as feature
            features[i, j] = np.mean(real_response)
    
    return features

def visualize_gabor_filters_and_responses(gabor_filters, sample_image):
    """Visualize some Gabor filters and their responses on a sample image"""
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle("Gabor Filters and Responses")
    
    sample_img = sample_image.reshape(14, 14)
    
    # Show first 4 filters and their responses
    for i in range(min(4, len(gabor_filters))):
        kernel, freq, theta = gabor_filters[i]
        
        # Show kernel real part
        axes[0, i].imshow(kernel.real, cmap='gray')
        axes[0, i].set_title(f'Filter {i+1}\nf={freq:.1f}, θ={theta*180/np.pi:.0f}°')
        axes[0, i].axis('off')
        
        # Show kernel imaginary part
        axes[1, i].imshow(kernel.imag, cmap='gray')
        axes[1, i].set_title(f'Imaginary part')
        axes[1, i].axis('off')
        
        # Show response
        real_response, _ = gabor(sample_img, frequency=freq, theta=theta, bandwidth=1)
        axes[2, i].imshow(real_response, cmap='gray')
        axes[2, i].set_title(f'Response (mean={np.mean(real_response):.3f})')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("gabor_filters_visualization.png", dpi=200)
    plt.show()

def main():
    print("Loading MNIST dataset...")
    # Load all 70k
    ds = fetch_openml('mnist_784', as_frame=False)
    X_all = ds.data.astype(np.float32)
    y_all = ds.target.astype(np.int64)
    
    # Sample 100 per class for training and validation (200 samples per class total)
    print("Sampling dataset...")
    X_sample, y_sample = sample_per_class(X_all, y_all, per_class=200)
    
    # Split into train/val (100 samples per class each)
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.5, random_state=42, stratify=y_sample
    )
    
    print("Before Downsample")
    print("Train:", X_train.shape, "Val:", X_val.shape)
    
    # Downsample to 14x14
    X_train_ds = downsample_blockavg(X_train)
    X_val_ds = downsample_blockavg(X_val)
    
    print("After Downsample")
    print("Train:", X_train_ds.shape, "Val:", X_val_ds.shape)
    
    # Create Gabor filter bank
    gabor_filters = create_gabor_filters()
    
    # Visualize filters and responses
    print("Visualizing Gabor filters...")
    visualize_gabor_filters_and_responses(gabor_filters, X_train_ds[0])
    
    # Extract Gabor features
    print("Extracting training features...")
    start_time = time.time()
    X_train_gabor = extract_gabor_features(X_train_ds, gabor_filters)
    train_time = time.time() - start_time
    
    print("Extracting validation features...")
    start_time = time.time()
    X_val_gabor = extract_gabor_features(X_val_ds, gabor_filters)
    val_time = time.time() - start_time
    
    print(f"Feature extraction time: train={train_time:.1f}s, val={val_time:.1f}s")
    print(f"Gabor feature shape: {X_train_gabor.shape}")
    
    # Train SVM with Gabor features
    print("Training SVM with Gabor features...")
    clf_gabor = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
    clf_gabor.fit(X_train_gabor, y_train)
    
    # Evaluate
    y_pred_gabor = clf_gabor.predict(X_val_gabor)
    val_acc_gabor = accuracy_score(y_val, y_pred_gabor)
    val_err_gabor = 1.0 - val_acc_gabor
    
    print(f"Gabor SVM - Validation accuracy: {val_acc_gabor:.4f} (error: {val_err_gabor:.4f})")
    
    # Support Vector ratio
    sv_ratio_gabor = len(clf_gabor.support_) / len(X_train_gabor)
    print(f"Gabor SVM - Support vectors: {len(clf_gabor.support_)} / {len(X_train_gabor)} (ratio: {sv_ratio_gabor:.3f})")
    
    # Compare with baseline (raw pixel) SVM
    print("\nTraining baseline SVM with raw pixels...")
    clf_baseline = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
    clf_baseline.fit(X_train_ds, y_train)
    
    y_pred_baseline = clf_baseline.predict(X_val_ds)
    val_acc_baseline = accuracy_score(y_val, y_pred_baseline)
    val_err_baseline = 1.0 - val_acc_baseline
    
    print(f"Baseline SVM - Validation accuracy: {val_acc_baseline:.4f} (error: {val_err_baseline:.4f})")
    
    sv_ratio_baseline = len(clf_baseline.support_) / len(X_train_ds)
    print(f"Baseline SVM - Support vectors: {len(clf_baseline.support_)} / {len(X_train_ds)} (ratio: {sv_ratio_baseline:.3f})")
    
    # Comparison
    print(f"\nImprovement: {val_acc_gabor - val_acc_baseline:.4f} accuracy points")
    
    # Confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gabor confusion matrix
    cm_gabor = confusion_matrix(y_val, y_pred_gabor, labels=np.arange(10))
    disp_gabor = ConfusionMatrixDisplay(confusion_matrix=cm_gabor, display_labels=np.arange(10))
    disp_gabor.plot(ax=ax1, values_format='d', colorbar=False)
    ax1.set_title(f"Gabor SVM (Acc: {val_acc_gabor:.3f})")
    
    # Baseline confusion matrix
    cm_baseline = confusion_matrix(y_val, y_pred_baseline, labels=np.arange(10))
    disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=np.arange(10))
    disp_baseline.plot(ax=ax2, values_format='d', colorbar=False)
    ax2.set_title(f"Baseline SVM (Acc: {val_acc_baseline:.3f})")
    
    plt.tight_layout()
    plt.savefig("gabor_vs_baseline_confusion.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()