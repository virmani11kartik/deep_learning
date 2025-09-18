from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
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

def create_gabor_filter_bank(theta_vals=None, freq_vals=None, bandwidth_vals=None):
    """Create a comprehensive Gabor filter bank"""
    if theta_vals is None:
        theta_vals = np.arange(0, np.pi, np.pi/4)  # [0, π/4, π/2, 3π/4]
    if freq_vals is None:
        freq_vals = np.arange(0.05, 0.5, 0.15)    # [0.05, 0.2, 0.35]
    if bandwidth_vals is None:
        bandwidth_vals = np.arange(0.3, 1, 0.3)   # [0.3, 0.6, 0.9]
    
    filters = []
    
    for freq in freq_vals:
        for theta in theta_vals:
            for bandwidth in bandwidth_vals:
                kernel = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)
                filters.append({
                    'kernel': kernel,
                    'freq': freq, 
                    'theta': theta, 
                    'bandwidth': bandwidth
                })
    
    print(f"Created filter bank with {len(filters)} Gabor filters")
    print(f"Orientations: {len(theta_vals)} ({theta_vals * 180/np.pi}°)")
    print(f"Frequencies: {len(freq_vals)} ({freq_vals})")
    print(f"Bandwidths: {len(bandwidth_vals)} ({bandwidth_vals})")
    
    return filters

def visualize_filter_bank(gabor_filters, save_name="gabor_filter_bank.png"):
    """Visualize the entire Gabor filter bank"""
    n_filters = len(gabor_filters)
    
    # Arrange in a grid - try to make it roughly square
    n_rows = int(np.sqrt(n_filters))
    n_cols = int(np.ceil(n_filters / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    fig.suptitle(f"Gabor Filter Bank ({n_filters} filters)", fontsize=16)
    
    axes = axes.flatten() if n_filters > 1 else [axes]
    
    for i, filter_info in enumerate(gabor_filters):
        kernel = filter_info['kernel']
        freq = filter_info['freq']
        theta = filter_info['theta']
        bandwidth = filter_info['bandwidth']
        
        # Show real part of the kernel
        axes[i].imshow(kernel.real, cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i].set_title(f'f={freq:.2f}, θ={theta*180/np.pi:.0f}°, b={bandwidth:.1f}', 
                         fontsize=8)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.show()

def extract_full_gabor_features(X, gabor_filters, method='all_pixels'):
    """Extract comprehensive Gabor filter responses"""
    n_samples = X.shape[0]
    n_filters = len(gabor_filters)
    
    if method == 'all_pixels':
        # Use all pixel responses - creates very high dimensional features
        features = np.zeros((n_samples, n_filters * 196))  # 14x14 = 196 pixels per filter
        feature_dim_per_filter = 196
    else:
        # Use statistical features (mean, std, etc.)
        features = np.zeros((n_samples, n_filters * 4))  # mean, std, min, max per filter
        feature_dim_per_filter = 4
    
    print(f"Extracting Gabor features using '{method}' method...")
    print(f"Feature dimensions: {features.shape}")
    
    for i in range(n_samples):
        if i % 200 == 0:
            print(f"Processing image {i}/{n_samples}")
            
        # Reshape to 14x14 image
        image = X[i].reshape(14, 14)
        
        for j, filter_info in enumerate(gabor_filters):
            freq = filter_info['freq']
            theta = filter_info['theta']
            bandwidth = filter_info['bandwidth']
            
            # Apply Gabor filter - use only real part
            real_response, _ = gabor(image, frequency=freq, theta=theta, bandwidth=bandwidth)
            
            if method == 'all_pixels':
                # Use all pixel responses
                start_idx = j * feature_dim_per_filter
                end_idx = start_idx + feature_dim_per_filter
                features[i, start_idx:end_idx] = real_response.flatten()
            else:
                # Use statistical features
                start_idx = j * feature_dim_per_filter
                features[i, start_idx] = np.mean(real_response)
                features[i, start_idx + 1] = np.std(real_response)
                features[i, start_idx + 2] = np.min(real_response)
                features[i, start_idx + 3] = np.max(real_response)
    
    return features

def apply_pca(X_train, X_val, n_components=None, variance_threshold=0.95):
    """Apply PCA to reduce dimensionality"""
    if n_components is None:
        # Choose components to retain specified variance
        pca_temp = PCA()
        pca_temp.fit(X_train)
        cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
    
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA retained {explained_var:.3f} of total variance")
    
    return X_train_pca, X_val_pca, pca

def compare_different_filter_banks():
    """Compare different filter bank configurations"""
    
    # Different configurations to test
    configs = [
        {
            'name': 'Small Bank (12 filters)',
            'theta': np.arange(0, np.pi, np.pi/4),      # 4 orientations
            'freq': [0.1, 0.2, 0.3],                    # 3 frequencies
            'bandwidth': [1.0]                          # 1 bandwidth
        },
        {
            'name': 'Standard Bank (36 filters)',
            'theta': np.arange(0, np.pi, np.pi/4),      # 4 orientations  
            'freq': np.arange(0.05, 0.5, 0.15),        # 3 frequencies
            'bandwidth': np.arange(0.3, 1, 0.3)        # 3 bandwidths
        },
        {
            'name': 'Large Bank (80 filters)',
            'theta': np.arange(0, np.pi, np.pi/8),      # 8 orientations
            'freq': np.arange(0.05, 0.5, 0.1),         # 5 frequencies  
            'bandwidth': [0.5, 1.0]                     # 2 bandwidths
        }
    ]
    
    return configs

def main():
    print("Loading MNIST dataset...")
    # Load all 70k
    ds = fetch_openml('mnist_784', as_frame=False)
    X_all = ds.data.astype(np.float32)
    y_all = ds.target.astype(np.int64)
    
    # Sample 100 per class for training and validation
    print("Sampling dataset...")
    X_sample, y_sample = sample_per_class(X_all, y_all, per_class=200)
    
    # Split into train/val (100 samples per class each)
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.5, random_state=42, stratify=y_sample
    )
    
    print("Dataset shapes:")
    print("Train:", X_train.shape, "Val:", X_val.shape)
    
    # Downsample to 14x14
    X_train_ds = downsample_blockavg(X_train)
    X_val_ds = downsample_blockavg(X_val)
    
    print("After downsampling:")
    print("Train:", X_train_ds.shape, "Val:", X_val_ds.shape)
    
    # Test different filter bank configurations
    configs = compare_different_filter_banks()
    results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        # Create filter bank
        gabor_filters = create_gabor_filter_bank(
            theta_vals=config['theta'],
            freq_vals=config['freq'], 
            bandwidth_vals=config['bandwidth']
        )
        
        # Visualize filter bank
        visualize_filter_bank(gabor_filters, f"filter_bank_{len(gabor_filters)}.png")
        
        # Extract features with statistical method first (lower dimension)
        print("\n--- Using Statistical Features ---")
        X_train_gabor_stats = extract_full_gabor_features(X_train_ds, gabor_filters, method='stats')
        X_val_gabor_stats = extract_full_gabor_features(X_val_ds, gabor_filters, method='stats')
        
        # Train SVM
        print("Training SVM with statistical Gabor features...")
        clf_stats = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
        clf_stats.fit(X_train_gabor_stats, y_train)
        
        # Evaluate
        y_pred_stats = clf_stats.predict(X_val_gabor_stats)
        val_acc_stats = accuracy_score(y_val, y_pred_stats)
        
        print(f"Statistical features - Validation accuracy: {val_acc_stats:.4f}")
        
        # Extract full pixel features (high dimension)
        print("\n--- Using All-Pixel Features ---")
        X_train_gabor_full = extract_full_gabor_features(X_train_ds, gabor_filters, method='all_pixels')
        X_val_gabor_full = extract_full_gabor_features(X_val_ds, gabor_filters, method='all_pixels')
        
        # Apply PCA for high-dimensional case
        if X_train_gabor_full.shape[1] > 1000:
            print("High dimensionality detected - applying PCA...")
            X_train_pca, X_val_pca, pca = apply_pca(X_train_gabor_full, X_val_gabor_full, 
                                                   variance_threshold=0.95)
        else:
            X_train_pca, X_val_pca = X_train_gabor_full, X_val_gabor_full
            pca = None
        
        # Train SVM with PCA features
        print("Training SVM with full Gabor features (after PCA)...")
        clf_full = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
        clf_full.fit(X_train_pca, y_train)
        
        # Evaluate
        y_pred_full = clf_full.predict(X_val_pca)
        val_acc_full = accuracy_score(y_val, y_pred_full)
        
        print(f"Full features (PCA) - Validation accuracy: {val_acc_full:.4f}")
        
        # Store results
        results.append({
            'name': config['name'],
            'n_filters': len(gabor_filters),
            'stats_accuracy': val_acc_stats,
            'full_accuracy': val_acc_full,
            'original_dim': X_train_gabor_full.shape[1],
            'pca_dim': X_train_pca.shape[1] if pca else X_train_gabor_full.shape[1]
        })
    
    # Compare with baseline
    print(f"\n{'='*60}")
    print("BASELINE: Raw Pixel SVM")
    print(f"{'='*60}")
    
    clf_baseline = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
    clf_baseline.fit(X_train_ds, y_train)
    y_pred_baseline = clf_baseline.predict(X_val_ds)
    val_acc_baseline = accuracy_score(y_val, y_pred_baseline)
    
    print(f"Baseline (raw pixels) - Validation accuracy: {val_acc_baseline:.4f}")
    
    # Summary results
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Method':<25} {'Filters':<10} {'Orig Dim':<10} {'PCA Dim':<10} {'Stats Acc':<12} {'Full Acc':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<25} {result['n_filters']:<10} {result['original_dim']:<10} "
              f"{result['pca_dim']:<10} {result['stats_accuracy']:<12.4f} {result['full_accuracy']:<12.4f}")
    
    print(f"{'Baseline (raw pixels)':<25} {'N/A':<10} {'196':<10} {'196':<10} {'N/A':<12} {val_acc_baseline:<12.4f}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: max(x['stats_accuracy'], x['full_accuracy']))
    best_acc = max(best_result['stats_accuracy'], best_result['full_accuracy'])
    improvement = best_acc - val_acc_baseline
    
    print(f"\nBest configuration: {best_result['name']}")
    print(f"Best accuracy: {best_acc:.4f} (improvement: {improvement:+.4f})")

if __name__ == "__main__":
    main()