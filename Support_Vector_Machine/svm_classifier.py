from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

RNG = np.random.RandomState(42)

def downsample_blockavg(X28x28):
    N = X28x28.shape[0]
    A = X28x28.reshape(N, 28, 28)
    B = A.reshape(N, 14, 2, 14, 2).mean(axis=(2, 4))
    return B.reshape(N, 196).astype(np.float32)

def sample_per_class(X, y, per_class=1000):
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

def main():
     # 1) Load all 70k
    ds = fetch_openml('mnist_784', as_frame=False)
    X_all = ds.data.astype(np.float32)
    y_all = ds.target.astype(np.int64)

    X_10k, y_10k = sample_per_class(X_all, y_all, per_class=1000)

    X_train, X_test, y_train, y_test = train_test_split(
        X_10k, y_10k, test_size=0.2, random_state=42, stratify=y_10k
    )
    print("Before Downsample")
    print("Train:", X_train.shape, "Val:", X_test.shape)

    X_train_ds = downsample_blockavg(X_train)
    X_test_ds = downsample_blockavg(X_test)

    print("After Downsample")
    print("Train:", X_train_ds.shape, "Val:", X_test_ds.shape)

    clf = svm.SVC(C=0.3, kernel='rbf', gamma='scale', verbose=True)
    clf.fit(X_train_ds, y_train)

    ## Validation error
    y_pred = clf.predict(X_test_ds)
    val_acc = accuracy_score(y_test,y_pred)
    val_err = 1.0 - val_acc
    print(f"Validation accuracy: {val_acc:.4f}  (error: {val_err:.4f})")

    ## Support Vector ratio

    sv_ratio = len(clf.support_) / len(X_train_ds)
    print(f"Support vectors: {len(clf.support_)} / {len(X_train_ds)}  (ratio: {sv_ratio:.3f})")

    test_acc = accuracy_score(y_test,y_pred)
    test_err = 1.0 - test_acc
    print(f"Test accuracy: {test_acc:.4f}  (error: {test_err:.4f})")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(values_format='d')      
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.show() 

    # try:
    #     X_test_ds = downsample_blockavg((X_test / 255.0).astype(np.float32))
    #     y_test_pred = clf.predict(X_test_ds)
    #     test_acc = accuracy_score(y_test, y_test_pred)
    #     test_err = 1.0 - test_acc
    #     print(f"Test accuracy: {test_acc:.4f}  (error: {test_err:.4f})")
    #     cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(10))
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    #     plt.figure()
    #     disp.plot(values_format='d')      
    #     plt.title("Confusion Matrix (Test Set)")
    #     plt.tight_layout()
    #     plt.savefig("confusion_matrix.png", dpi=200)
    #     plt.show() 
    #     print("Saved confusion matrix to: confusion_matrix.png")
    # except NameError:
    #     print("No X_test/y_test detected — skipping test evaluation.")

if __name__ == "__main__":
    main()